"""Beam comparison: the supplied evaluator is required; no substitute equations."""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import urllib.request
from xml.sax.saxutils import escape

START = (50.0, 80.0)
GPT_MODEL = 'gpt-4o-mini-2024-07-18'
LLAMA_MODEL = 'llama3.2:1b'
TEMPERATURE = 0.7
PROPOSALS = 11
FIELDS = ['method', 'model', 'returned_model', 'run', 'iteration', 'width_mm',
          'height_mm', 'mass_kg', 'max_deflection_mm', 'max_von_mises_stress_MPa',
          'deflection_margin_mm', 'stress_margin_MPa', 'feasible',
          'feasible_with_tolerance', 'response_status', 'beam_evaluated',
          'repeated_design', 'best_feasible_mass_kg', 'elapsed_s', 'raw_response', 'error']


def validate_reply(raw, beam):
    """Strict JSON, exactly two finite numeric dimensions, within bounds. No repairs."""
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate JSON key')
            obj[key] = value
        return obj
    obj = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(obj, dict) or set(obj) != {'width_mm', 'height_mm'}:
        raise ValueError('Expected exactly width_mm and height_mm')
    values = (obj['width_mm'], obj['height_mm'])
    for value, limits in zip(values, [beam.WIDTH_BOUNDS_MM, beam.HEIGHT_BOUNDS_MM]):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Dimensions must be finite JSON numbers')
        if not limits[0] <= value <= limits[1]:
            raise ValueError('Dimension outside bounds')
    return values


def evaluate(beam, width, height):
    result = beam.evaluate_beam(width, height)
    keys = ['mass_kg', 'max_deflection_mm', 'max_von_mises_stress_MPa']
    out = {key: float(result[key]) for key in keys}
    if not all(math.isfinite(v) and v >= 0 for v in out.values()):
        raise ValueError('Unexpected evaluator outputs; check provided file and units')
    dm = beam.MAX_DEFLECTION_MM - out['max_deflection_mm']
    sm = beam.MAX_VON_MISES_STRESS_MPA - out['max_von_mises_stress_MPa']
    bounded = (beam.WIDTH_BOUNDS_MM[0] <= width <= beam.WIDTH_BOUNDS_MM[1]
               and beam.HEIGHT_BOUNDS_MM[0] <= height <= beam.HEIGHT_BOUNDS_MM[1])
    out.update(deflection_margin_mm=dm, stress_margin_MPa=sm,
               feasible=bool(bounded and dm >= 0 and sm >= 0),
               feasible_with_tolerance=bool(bounded and dm / beam.MAX_DEFLECTION_MM >= -1e-6
                                           and sm / beam.MAX_VON_MISES_STRESS_MPA >= -1e-6))
    return out


def make_messages(beam, history):
    system = ('You select beam designs in a black-box optimization experiment. '
              'Minimize mass_kg while satisfying all bounds and constraints. '
              'Use previous evaluation feedback. You have only 11 proposals after the starting design. '
              'Do not repeat a tested design. Return only one JSON object with exactly '
              'two numeric keys: width_mm and height_mm. No markdown, explanation, or extra keys. '
              'Invalid replies consume a proposal. No corrections or replacement designs are made.')
    task = {'objective': 'minimize mass_kg', 'units': 'dimensions in mm',
            'width_bounds_mm': list(beam.WIDTH_BOUNDS_MM),
            'height_bounds_mm': list(beam.HEIGHT_BOUNDS_MM),
            'max_deflection_mm': beam.MAX_DEFLECTION_MM,
            'max_von_mises_stress_MPa': beam.MAX_VON_MISES_STRESS_MPA,
            'proposal_number': len(history), 'previous_attempts': history}
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(task, allow_nan=False)}]


def request_model(provider, messages):
    if provider == 'openai':
        from openai import OpenAI
        with OpenAI(max_retries=0, timeout=60) as client:
            response = client.chat.completions.create(
                model=GPT_MODEL, messages=messages, temperature=TEMPERATURE, top_p=1.0,
                max_completion_tokens=160)
        return response.choices[0].message.content or '', response.model
    payload = {'model': LLAMA_MODEL, 'messages': messages, 'stream': False,
               'options': {'temperature': TEMPERATURE, 'top_p': 1.0,
                           'num_predict': 160, 'num_ctx': 8192}}
    url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434').rstrip('/') + '/api/chat'
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    return data['message']['content'], data.get('model', LLAMA_MODEL)


def run_agent(beam, provider, run, record, caller=request_model):
    model = GPT_MODEL if provider == 'openai' else LLAMA_MODEL
    started = time.perf_counter()
    history, seen, rows = [], set(), []
    best = None
    for iteration in range(PROPOSALS + 1):
        row = dict(method='AI', model=model, run=run, iteration=iteration,
                   beam_evaluated=False, repeated_design=False)
        dimensions = None
        if iteration == 0:
            dimensions = START
            row['response_status'] = 'starting_design'
        else:
            try:
                raw, returned = caller(provider, make_messages(beam, history))
                row.update(raw_response=raw, returned_model=returned)
            except Exception as exc:
                # Do not include provider error bodies: they might contain credentials.
                row.update(response_status='service_error', error=type(exc).__name__)
            else:
                try:
                    dimensions = validate_reply(raw, beam)
                    row['response_status'] = 'valid'
                except (ValueError, TypeError, OverflowError) as exc:
                    row.update(response_status='invalid_reply', error=str(exc))
        if dimensions is not None:
            width, height = dimensions
            row.update(width_mm=width, height_mm=height,
                       repeated_design=dimensions in seen, beam_evaluated=True)
            seen.add(dimensions)
            try:
                row.update(evaluate(beam, width, height))
            except Exception as exc:
                row.update(response_status='evaluation_error', error=type(exc).__name__)
            if row.get('feasible'):
                best = min(best, row['mass_kg']) if best is not None else row['mass_kg']
        row.update(best_feasible_mass_kg=best, elapsed_s=time.perf_counter() - started)
        rows.append(row)
        record(row)
        history.append({k: v for k, v in row.items()
                        if k not in ('method', 'model', 'returned_model', 'run', 'elapsed_s', 'raw_response')})
        print(f'{model} run {run}, proposal {iteration}: {row["response_status"]}; best={best}', flush=True)
    return rows, {'method': 'AI', 'model': model, 'run': run,
                  'elapsed_s': time.perf_counter() - started}


def run_scipy(beam, record):
    from scipy.optimize import minimize
    started = time.perf_counter()
    cache, rows = {}, []
    best = None
    def get(x):
        nonlocal best
        key = tuple(float(v) for v in x)
        if key not in cache:
            out = evaluate(beam, *key)
            cache[key] = out
            if out['feasible']:
                best = min(best, out['mass_kg']) if best is not None else out['mass_kg']
            row = dict(method='SciPy', model='SLSQP', run=1, iteration=len(rows),
                       width_mm=key[0], height_mm=key[1], **out,
                       response_status='evaluated', beam_evaluated=True, repeated_design=False,
                       best_feasible_mass_kg=best, elapsed_s=time.perf_counter() - started)
            rows.append(row)
            record(row)
        return cache[key]
    baseline = get(START)
    expected = {'mass_kg': 31.4, 'max_deflection_mm': 0.48828125,
                'max_von_mises_stress_MPa': 46.875}
    if any(not math.isclose(baseline[k], v, rel_tol=.02) for k, v in expected.items()):
        raise ValueError('Starting outputs disagree with worksheet. Inspect beam_model.py units before running AI.')
    result = minimize(lambda x: get(x)['mass_kg'], START, method='SLSQP',
                      bounds=[beam.WIDTH_BOUNDS_MM, beam.HEIGHT_BOUNDS_MM],
                      constraints=[
                          {'type': 'ineq', 'fun': lambda x: 1 - get(x)['max_deflection_mm'] / beam.MAX_DEFLECTION_MM},
                          {'type': 'ineq', 'fun': lambda x: 1 - get(x)['max_von_mises_stress_MPa'] / beam.MAX_VON_MISES_STRESS_MPA}],
                      options={'maxiter': 200, 'ftol': 1e-9})
    final = get(result.x)
    info = {'method': 'SciPy', 'model': 'SLSQP', 'run': 1,
            'success': bool(result.success), 'message': str(result.message),
            'width_mm': float(result.x[0]), 'height_mm': float(result.x[1]),
            'final_outputs': final, 'solver_nfev': int(result.nfev),
            'elapsed_s': time.perf_counter() - started}
    return rows, info


def summarize(rows, timings):
    summaries = []
    for info in timings:
        group = [r for r in rows if (r['model'], r['run']) == (info['model'], info['run'])]
        feasible = [r for r in group if r.get('feasible')]
        best = min(feasible, key=lambda r: r['mass_kg']) if feasible else None
        summaries.append(dict(info, best=best, evaluations=sum(r['beam_evaluated'] for r in group),
                              invalid=sum(r['response_status'] == 'invalid_reply' for r in group),
                              service_errors=sum(r['response_status'] == 'service_error' for r in group),
                              evaluation_errors=sum(r['response_status'] == 'evaluation_error' for r in group),
                              repeats=sum(r['repeated_design'] for r in group)))
    return summaries


def write_report(prefix, rows, summaries, metadata, repo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for info in summaries:
        if info['method'] != 'AI':
            continue
        group = [r for r in rows if r['model'] == info['model'] and r['run'] == info['run']]
        ax.step([r['iteration'] for r in group],
                [r['best_feasible_mass_kg'] if r['best_feasible_mass_kg'] is not None else float('nan') for r in group],
                where='post', marker='.', label=f'{info["model"]}, run {info["run"]}')
    numerical = summaries[0]['best']
    if numerical:
        ax.axhline(numerical['mass_kg'], linestyle='--', color='black', label='SciPy best feasible')
    ax.set(xlabel='Proposal number (0 = common start; failures retain incumbent)',
           ylabel='Best feasible mass (kg)', xticks=range(12))
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plot = str(prefix) + '_history.png'
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    from matplotlib import font_manager
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('BeamSans', font_manager.findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('BeamBold', font_manager.findfont(
        font_manager.FontProperties(family='DejaVu Sans', weight='bold'))))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'BeamBold' if style.name in ('Title', 'Heading1', 'Heading2') else 'BeamSans'
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    story = []
    def p(text, style='BodyText'):
        story.append(Paragraph(escape(text), styles[style]))
        story.append(Spacer(1, 8))
    p('Beam Design and AI Model Comparison', 'Title')
    p('Jacob Coone | Individual Project Part 2')
    p('GitHub repository: ' + (repo or 'NOT PROVIDED - add before submission'))
    p('Method', 'Heading2')
    p('The goal was to find the lightest feasible beam using the supplied beam_model.py. '
      'Every run started at 50 x 80 mm. SciPy used SLSQP. Each AI model had two independent runs '
      'with 11 proposals each. Both models received the same prompt structure, validation rule, '
      'and their own previous results. AI designs were never chosen by a numerical optimizer.')
    baseline = rows[0]
    p(f'The common start returned {baseline["mass_kg"]:.4f} kg, '
      f'{baseline["max_deflection_mm"]:.4f} mm deflection, and '
      f'{baseline["max_von_mises_stress_MPa"]:.4f} MPa stress. Feasible: {baseline["feasible"]}.')
    p('Results', 'Heading2')
    data = [['Method / run', 'Best kg', 'W mm', 'H mm', 'Defl. margin\nmm', 'Stress margin\nMPa', 'Evals', 'Invalid', 'Time s']]
    for s in summaries:
        b = s['best']
        label = 'SLSQP' if s['method'] == 'SciPy' else ('GPT mini' if s['model'] == GPT_MODEL else 'Llama 1B') + f' / {s["run"]}'
        nums = [f'{b[k]:.4f}' for k in ['mass_kg', 'width_mm', 'height_mm', 'deflection_margin_mm', 'stress_margin_MPa']] if b else ['N/A'] * 5
        data.append([label] + nums + [str(s['evaluations']), str(s['invalid']), f'{s["elapsed_s"]:.2f}'])
    table = Table(data, colWidths=[76, 48, 47, 47, 63, 67, 33, 37, 44])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6edf5')),
                               ('FONTNAME', (0, 0), (-1, -1), 'BeamSans'),
                               ('FONTNAME', (0, 0), (-1, 0), 'BeamBold'),
                               ('FONTSIZE', (0, 0), (-1, -1), 7), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                               ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey)]))
    story.append(table)
    story.append(Spacer(1, 12))
    opt = summaries[0]
    p(f'Optimizer convergence: {opt["success"]}. {opt["message"]} '
      f'Returned design: {opt["width_mm"]:.6f} x {opt["height_mm"]:.6f} mm; '
      f'strict feasibility: {opt["final_outputs"]["feasible"]}; '
      f'feasible within normalized 1e-6 tolerance: {opt["final_outputs"]["feasible_with_tolerance"]}. '
      'The table uses strictly feasible evaluated designs, which can differ from the final solver point.')
    story.append(Image(plot, width=500, height=225))
    story.append(PageBreak())
    p('Comparison and run variation', 'Heading2')
    for model in [GPT_MODEL, LLAMA_MODEL]:
        pair = [s for s in summaries if s['model'] == model]
        for s in pair:
            group = [r for r in rows if r['model'] == model and r['run'] == s['run']]
            valid = [r for r in group if r.get('response_status') == 'valid' and 'mass_kg' in r]
            improvements, incumbent = 0, baseline['mass_kg']
            for r in valid:
                if r.get('feasible') and r['mass_kg'] < incumbent:
                    improvements += 1
                    incumbent = r['mass_kg']
            p(f'{model}, run {s["run"]}: {improvements} improvements to the best feasible mass, '
              f'{s["repeats"]} repeated designs, {s["invalid"]} invalid replies '
              f'(including {sum(r.get("error") == "Dimension outside bounds" for r in group)} out-of-bounds replies), '
              f'{s["service_errors"]} service failures, and {s["evaluation_errors"]} evaluator failures. '
              f'{sum(not r["feasible"] for r in valid)} evaluated proposals were infeasible. '
              + ('Sequential improvement is consistent with using feedback, but does not prove the model understood the beam.'
                 if improvements else 'No improvement over the starting design was observed in this run.'))
        bests = [s['best']['mass_kg'] for s in pair if s['best']]
        if len(bests) == 2:
            p(f'The two {model} runs differed by {abs(bests[0] - bests[1]):.4f} kg in best feasible mass. '
              + (f'Their best mass was {min(bests) - numerical["mass_kg"]:+.4f} kg relative to SciPy.' if numerical else 'SciPy had no strictly feasible evaluated design.'))
    p('Reproducibility and limitations', 'Heading2')
    p('Temperature was 0.7; output limit was 160 tokens. No explicit random seed was set. '
      'The exact requested models were ' + GPT_MODEL + ' and ' + LLAMA_MODEL + '. '
      'Returned model identifiers are logged. Ollama used an 8192-token context and top_p=1. '
      'Strict JSON validation rejected extra keys, strings, booleans, nonfinite values, duplicate keys, '
      'markdown and out-of-bounds dimensions. Invalid replies and service errors consumed a proposal '
      'without a beam evaluation. No reply was repaired or retried.')
    p('Only two runs per model were used, so the results are a small sample. '
      'Elapsed time includes service latency and local model loading, but excludes plotting and report generation. '
      'Software, hardware, and service differences affect timing. SLSQP caches exact repeated coordinates; '
      'its evaluation count is the actual number of black-box calls. AI repeats are evaluated and counted. '
      'Margins are limit minus output; negative margins are infeasible. '
      'The worksheet contains a meters/mm inconsistency; the example and named parameters use millimeters, '
      'so this experiment uses millimeters and checks the starting outputs. '
      'This is a static, ideal beam model, not an impact or local-contact stress model.')
    p('Software: ' + json.dumps(metadata['software'], sort_keys=True))
    p('Evaluator SHA-256: ' + metadata['beam_sha256'])
    p('Documentation: https://developers.openai.com/api/docs/models/gpt-4o-mini ; https://docs.ollama.com/api/chat')
    SimpleDocTemplate(str(prefix) + '.pdf', rightMargin=34, leftMargin=34,
                      topMargin=34, bottomMargin=34).build(story)
    return plot


def main(username='jcoone', repo='https://github.com/jakecoone/vip-beam-agent', output_dir='results'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Environment variables also work without a .env file.
    try:
        beam = importlib.import_module('beam_model')
    except ModuleNotFoundError as exc:
        raise SystemExit('Missing supplied beam_model.py. Download it from Brightspace and place it beside this script.') from exc
    if not username.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use your Purdue username, with letters/numbers/underscore/hyphen only')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f'{username}_beam_ai'
    all_rows, timings = [], []
    with open(str(prefix) + '.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        def record(row):
            all_rows.append(row)
            writer.writerow(row)
            stream.flush()
        _, info = run_scipy(beam, record)
        timings.append(info)
        for provider in ['openai', 'ollama']:
            for run in [1, 2]:
                _, info = run_agent(beam, provider, run, record)
                timings.append(info)
    software = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['scipy', 'numpy', 'openai', 'python-dotenv', 'matplotlib', 'reportlab']:
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            software[name] = 'not installed'
    metadata = {'software': software, 'beam_sha256': hashlib.sha256(Path(beam.__file__).read_bytes()).hexdigest(),
                'gpt_model': GPT_MODEL, 'llama_model': LLAMA_MODEL, 'temperature': TEMPERATURE,
                'repo': repo, 'start_mm': START, 'proposals_per_run': PROPOSALS}
    summaries = summarize(all_rows, timings)
    Path(str(prefix) + '_summary.json').write_text(json.dumps({'metadata': metadata, 'results': summaries}, indent=2))
    plot = write_report(prefix, all_rows, summaries, metadata, repo)
    print(f'Wrote {prefix}.csv, {prefix}.pdf, {plot} and summary JSON.')
    return all_rows, summaries, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', default='jcoone')
    parser.add_argument('--repo', default='https://github.com/jakecoone/vip-beam-agent')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    main(args.username, args.repo, args.output_dir)

"""Fetch pinned, owner-authorized model assets outside Git and verify ONNX contracts."""
import argparse, hashlib, json, os, pathlib
import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='private/models')
parser.add_argument('--download', action='store_true')
args = parser.parse_args()
root = pathlib.Path(args.directory)
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads(pathlib.Path('models/manifest.json').read_text())
report = {'scope': 'Exact SHA-256 + ONNX checker + CPU inference on zeros; not pose accuracy validation.', 'models': []}
for item in manifest['models']:
    path = root / item['name']
    if args.download and not path.exists():
        import gdown
        gdown.download(id=os.environ['DRIVE_' + item['id'].upper()], output=str(path), quiet=False)
    assert path.stat().st_size == item['bytes'], f'{path}: wrong file size'
    assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == item['sha256'], f'{path}: SHA-256 mismatch'
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    shape = [1, 3, 640, 640] if item['id'] == 'detector' else [1, 3, 384, 288]
    assert len(inputs) == 1 and inputs[0].type == 'tensor(float)'
    assert all(not isinstance(a, int) or a == b for a, b in zip(inputs[0].shape, shape)) and len(inputs[0].shape) == 4
    outputs = session.run(None, {inputs[0].name: np.zeros(shape, np.float32)})
    expected = [[1, 8400, 85]] if item['id'] == 'detector' else [[1, 133, 576], [1, 133, 768]]
    assert [list(v.shape) for v in outputs] == expected
    assert all(np.isfinite(v).all() for v in outputs)
    report['models'].append({**item, 'input': inputs[0].shape, 'outputs': [list(v.shape) for v in outputs], 'onnxCheck': 'passed', 'cpuInference': 'passed', 'opsets': [{ 'domain': o.domain, 'version': o.version } for o in graph.opset_import]})
    del graph, session, outputs
(root / 'compatibility.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

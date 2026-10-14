"""Generate CONSTANT TEST DOUBLES, never real pose estimators. Ignored private path."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
root=Path('private/test-models');root.mkdir(parents=True,exist_ok=True)
def write(name,shape,outputs):
    nodes=[];infos=[]
    for key,value in outputs.items():
        nodes.append(helper.make_node('Constant',[],[key],value=numpy_helper.from_array(value)))
        infos.append(helper.make_tensor_value_info(key,TensorProto.FLOAT,list(value.shape)))
    graph=helper.make_graph(nodes,'TEST_DOUBLE_NOT_TRAINED',[helper.make_tensor_value_info('input',TensorProto.FLOAT,shape)],infos)
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);m.ir_version=8
    onnx.checker.check_model(m);onnx.save(m,root/name)
a=np.zeros((1,8400,85),dtype=np.float32)
# First stride 8 cell at (40,30); one high-confidence person, centered in image.
i=30*80+40;a[0,i,:6]=[0,0,np.log(100/8),np.log(200/8),.99,.99]
write('TEST-ONLY-detector.onnx',[1,3,640,640],{'output':a})
x=np.zeros((1,133,576),dtype=np.float32);y=np.zeros((1,133,768),dtype=np.float32)
for j in range(133):x[0,j,250+j%20]=.95;y[0,j,300+j%30]=.95
write('TEST-ONLY-pose.onnx',[1,3,384,288],{'simcc_x':x,'simcc_y':y})
print('Generated test doubles in',root)

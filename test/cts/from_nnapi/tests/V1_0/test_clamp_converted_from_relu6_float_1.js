'use strict';
import * as utils from '../../../../utils.js';

describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test clamp converted from relu6_float_1 test', async function() {
    // Converted test case (from: V1_0/relu6_float_1.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Buffer = new Float32Array([-10.0, -0.5, 0.5, 10.0]);
    const expected = [0.0, 0.0, 0.5, 6.0];
    const interOut0 = builder.clamp(op1);
    const op2 = builder.clamp(interOut0, {minValue: builder.constant(0), maxValue: builder.constant(6)});
    const model = builder.createModel({op2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op2.buffer, expected, utils.atol, utils.rtol);
  });
});

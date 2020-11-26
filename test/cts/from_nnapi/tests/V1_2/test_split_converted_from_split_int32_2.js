'use strict';
import * as utils from '../../../../utils.js';

describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test split converted from split_int32_2 test', async function() {
    // Converted test case (from: V1_2/split_int32_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 3]});
    const input0Buffer = new Int32Array([1, 2, 3, 4, 5, 6]);
    const axis = 0;
    const num_splits = 2;
    const expected = [[1, 2, 3], [4, 5, 6]];
    const [output0, output1] = builder.split(input0, num_splits, {'axis': axis});
    const model = builder.createModel({output0, output1});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]].buffer, expected[i], 1e-5, 5.0 * 1.1920928955078125e-7);
    }
  });

  it('test split converted from split_int32_2_relaxed test', async function() {
    // Converted test case (from: V1_2/split_int32_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 3]});
    const input0Buffer = new Int32Array([1, 2, 3, 4, 5, 6]);
    const axis = 0;
    const num_splits = 2;
    const expected = [[1, 2, 3], [4, 5, 6]];
    const [output0, output1] = builder.split(input0, num_splits, {'axis': axis});
    const model = builder.createModel({output0, output1});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]].buffer, expected[i], 5.0 * 0.0009765625, 5.0 * 0.0009765625);
    }
  });
});
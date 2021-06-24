'use strict';
import * as utils from '../utils.js';

describe('test relu', function() {
  const context = navigator.ml.createContext();

  it('relu', async function() {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: [3, 4, 5]});
    const y = builder.relu(x);
    const graph = await builder.build({y});
    const inputs = {
      'x': {
        data: new Float32Array([
          -1.483762,   0.6447428,   -1.2266507,  -1.7132527,  0.9777725,
          -0.34438756, -0.99921757, -1.2882805,  1.3725083,   -0.06386258,
          -0.44738683, -0.6776338,  0.5027815,   -1.0428967,  -1.4220539,
          0.00880813,  -1.2053454,  1.1644533,   -1.6577007,  -0.33448243,
          0.69386536,  0.06171616,  -0.20644434, 1.0620342,   -0.8824057,
          -0.7676657,  0.7517342,   1.4035656,   -0.29105335, 0.18367627,
          1.3628657,   -0.39770076, -0.1550809,  -1.2575449,  0.5797014,
          -0.02414344, 0.9181723,   -1.1963434,  0.56652546,  -0.25052008,
          -0.02097719, -2.6274924,  0.7993208,   -0.31359985, 0.9019325,
          -0.02042965, 0.5222995,   1.3394557,   -1.0482218,  1.1774449,
          0.8999488,   -1.1143959,  1.0122099,   -0.48604885, -0.06009902,
          -0.1766853,  1.4515465,   -0.7182982,  2.0361354,   0.7899623,
        ]),
      },
    };
    const outputs = await graph.compute(inputs);
    utils.checkShape(outputs.y.dimensions, [3, 4, 5]);
    const expected = [
      0.,        0.6447428, 0.,         0.,         0.9777725, 0.,
      0.,        0.,        1.3725083,  0.,         0.,        0.,
      0.5027815, 0.,        0.,         0.00880813, 0.,        1.1644533,
      0.,        0.,        0.69386536, 0.06171616, 0.,        1.0620342,
      0.,        0.,        0.7517342,  1.4035656,  0.,        0.18367627,
      1.3628657, 0.,        0.,         0.,         0.5797014, 0.,
      0.9181723, 0.,        0.56652546, 0.,         0.,        0.,
      0.7993208, 0.,        0.9019325,  0.,         0.5222995, 1.3394557,
      0.,        1.1774449, 0.8999488,  0.,         1.0122099, 0.,
      0.,        0.,        1.4515465,  0.,         2.0361354, 0.7899623,
    ];
    utils.checkValue(outputs.y.data, expected);
  });

  it('relu with preallocated buffer', async function() {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: [3, 4, 5]});
    const y = builder.relu(x);
    const graph = await builder.build({y});
    const inputs = {
      'x': {
        data: new Float32Array([
          -1.483762,   0.6447428,   -1.2266507,  -1.7132527,  0.9777725,
          -0.34438756, -0.99921757, -1.2882805,  1.3725083,   -0.06386258,
          -0.44738683, -0.6776338,  0.5027815,   -1.0428967,  -1.4220539,
          0.00880813,  -1.2053454,  1.1644533,   -1.6577007,  -0.33448243,
          0.69386536,  0.06171616,  -0.20644434, 1.0620342,   -0.8824057,
          -0.7676657,  0.7517342,   1.4035656,   -0.29105335, 0.18367627,
          1.3628657,   -0.39770076, -0.1550809,  -1.2575449,  0.5797014,
          -0.02414344, 0.9181723,   -1.1963434,  0.56652546,  -0.25052008,
          -0.02097719, -2.6274924,  0.7993208,   -0.31359985, 0.9019325,
          -0.02042965, 0.5222995,   1.3394557,   -1.0482218,  1.1774449,
          0.8999488,   -1.1143959,  1.0122099,   -0.48604885, -0.06009902,
          -0.1766853,  1.4515465,   -0.7182982,  2.0361354,   0.7899623,
        ]),
      },
    };
    const outputBuffer = new Float32Array(60);
    const outputs = await graph.compute(inputs, {y: outputBuffer});
    utils.checkShape(outputs.y.dimensions, [3, 4, 5]);
    const expected = [
      0.,        0.6447428, 0.,         0.,         0.9777725, 0.,
      0.,        0.,        1.3725083,  0.,         0.,        0.,
      0.5027815, 0.,        0.,         0.00880813, 0.,        1.1644533,
      0.,        0.,        0.69386536, 0.06171616, 0.,        1.0620342,
      0.,        0.,        0.7517342,  1.4035656,  0.,        0.18367627,
      1.3628657, 0.,        0.,         0.,         0.5797014, 0.,
      0.9181723, 0.,        0.56652546, 0.,         0.,        0.,
      0.7993208, 0.,        0.9019325,  0.,         0.5222995, 1.3394557,
      0.,        1.1774449, 0.8999488,  0.,         1.0122099, 0.,
      0.,        0.,        1.4515465,  0.,         2.0361354, 0.7899623,
    ];
    utils.checkValue(outputs.y.data, expected);
  });
});

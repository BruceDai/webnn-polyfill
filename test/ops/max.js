'use strict';
import * as utils from '../utils.js';

describe('test max', function() {
  const context = navigator.ml.createContext();

  it('max', function() {
    const builder = new MLGraphBuilder(context);
    const a = builder.input('a', {type: 'float32', dimensions: [3, 4, 5]});
    const b = builder.input('b', {type: 'float32', dimensions: [3, 4, 5]});
    const c = builder.max(a, b);
    const graph = builder.build({c});
    const inputs = {
      'a': new Float32Array([
        0.54270846, 0.3356357,  0.04034169,  1.6710619,   -1.0029255,
        1.4024457,  -0.5183214, -1.5897884,  0.16786452,  -0.92690915,
        -0.8761584, 1.8612522,  0.2960607,   0.11604685,  0.2686291,
        -0.5718065, 0.4856556,  -1.2307562,  -1.7977105,  -1.1370704,
        1.0383102,  -1.0015849, -1.367141,   0.32427165,  1.2968429,
        1.3039074,  -0.6295407, 1.1250858,   1.0206878,   -0.769062,
        0.96548617, 1.9100864,  2.1261373,   0.8835118,   -0.66880584,
        0.9088927,  1.8120629,  -0.25648043, 0.15793198,  -1.5175776,
        0.08734574, 0.9441932,  -1.0558261,  0.1276651,   -2.9616504,
        2.1102998,  0.58067006, -0.7349921,  -0.28586444, -0.92654175,
        -0.507083,  -1.8776977, 0.57921827,  1.460351,    1.4930215,
        -0.757663,  1.0773797,  -1.1858964,  -0.5337765,  0.27636543,
      ]),
      'b': new Float32Array([
        -0.00724315, -1.4088361,  0.17466596,  1.1395162,   1.3720452,
        -0.35610083, -0.5597993,  -0.26632488, -0.31922337, -0.2980101,
        0.12268824,  -1.1521344,  -1.0502838,  0.85281086,  -0.83374727,
        0.00551354,  0.08081324,  -0.13748081, 0.59067047,  -0.20894054,
        -0.9008378,  -0.06121079, -1.8927814,  -0.5113896,  2.0618987,
        -0.09704968, 1.9003097,   -0.27883208, -0.9971944,  -1.0472671,
        0.995112,    0.83037376,  1.5058613,   0.51366556,  0.4476341,
        1.0389726,   -0.04508441, -0.2180115,  0.3973936,   0.58917326,
        2.3834932,   0.71679467,  0.06214673,  -0.09415992, 0.9173279,
        0.55409455,  0.6537859,   -1.1739589,  1.1591603,   0.5907742,
        -1.0454807,  -0.8065648,  2.0162134,   -0.30215183, 0.67375183,
        1.6682644,   -2.916385,   0.43166366,  -0.7290503,  0.11509943,
      ]),
    };
    const outputs = {c: new Float32Array(utils.sizeOfShape([3, 4, 5]))};
    graph.compute(inputs, outputs);
    const expected = [
      0.54270846, 0.3356357,   0.17466596, 1.6710619,   1.3720452,  1.4024457,
      -0.5183214, -0.26632488, 0.16786452, -0.2980101,  0.12268824, 1.8612522,
      0.2960607,  0.85281086,  0.2686291,  0.00551354,  0.4856556,  -0.13748081,
      0.59067047, -0.20894054, 1.0383102,  -0.06121079, -1.367141,  0.32427165,
      2.0618987,  1.3039074,   1.9003097,  1.1250858,   1.0206878,  -0.769062,
      0.995112,   1.9100864,   2.1261373,  0.8835118,   0.4476341,  1.0389726,
      1.8120629,  -0.2180115,  0.3973936,  0.58917326,  2.3834932,  0.9441932,
      0.06214673, 0.1276651,   0.9173279,  2.1102998,   0.6537859,  -0.7349921,
      1.1591603,  0.5907742,   -0.507083,  -0.8065648,  2.0162134,  1.460351,
      1.4930215,  1.6682644,   1.0773797,  0.43166366,  -0.5337765, 0.27636543,
    ];
    utils.checkValue(outputs.c, expected);
  });

  it('max broadcast', function() {
    const builder = new MLGraphBuilder(context);
    const a = builder.input('a', {type: 'float32', dimensions: [3, 4, 5]});
    const b = builder.input('b', {type: 'float32', dimensions: [5]});
    const c = builder.max(a, b);
    const graph = builder.build({c});
    const inputs = {
      'a': new Float32Array([
        -0.78042406, -0.18523395, -0.12612817, -0.24858657, 0.36215156,
        -0.41349608, 1.540389,    1.9143543,   0.4806893,   0.0123093,
        1.2142435,   -0.57421523, -2.1229508,  1.1247561,   0.11206079,
        0.5191412,   -0.2109448,  -0.97485703, 0.6992101,   1.0161952,
        -0.19765139, 0.34198883,  -0.24741505, 1.5920583,   0.56292,
        0.09105966,  0.82438636,  -0.2996084,  -0.97498095, 1.9305013,
        1.4938543,   0.01099077,  0.7837045,   0.6621192,   0.9520401,
        -0.63094735, -1.4202772,  2.6008792,   -0.3047365,  -0.58313465,
        -0.37956452, -0.14322324, -1.2261407,  -1.1514657,  -0.28318587,
        -0.06985976, 0.48337674,  0.99673945,  -0.54980195, -1.7497128,
        0.62820524,  1.0456259,   0.16508068,  0.5966878,   0.7607826,
        0.9664813,   -0.13389224, -0.5757679,  0.38655168,  -0.39935285,
      ]),
      'b': new Float32Array([
        0.67538136,
        0.3535401,
        1.0303422,
        -0.50294054,
        -0.25600532,
      ]),
    };
    const outputs = {c: new Float32Array(utils.sizeOfShape([3, 4, 5]))};
    graph.compute(inputs, outputs);
    const expected = [
      0.67538136, 0.3535401,  1.0303422, -0.24858657, 0.36215156,
      0.67538136, 1.540389,   1.9143543, 0.4806893,   0.0123093,
      1.2142435,  0.3535401,  1.0303422, 1.1247561,   0.11206079,
      0.67538136, 0.3535401,  1.0303422, 0.6992101,   1.0161952,
      0.67538136, 0.3535401,  1.0303422, 1.5920583,   0.56292,
      0.67538136, 0.82438636, 1.0303422, -0.50294054, 1.9305013,
      1.4938543,  0.3535401,  1.0303422, 0.6621192,   0.9520401,
      0.67538136, 0.3535401,  2.6008792, -0.3047365,  -0.25600532,
      0.67538136, 0.3535401,  1.0303422, -0.50294054, -0.25600532,
      0.67538136, 0.48337674, 1.0303422, -0.50294054, -0.25600532,
      0.67538136, 1.0456259,  1.0303422, 0.5966878,   0.7607826,
      0.9664813,  0.3535401,  1.0303422, 0.38655168,  -0.25600532,
    ];
    utils.checkValue(outputs.c, expected);
  });
});

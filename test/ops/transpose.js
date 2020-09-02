describe('test transpose', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  async function checkTranspose(
      inputShape, inputData, expected, permutation = undefined) {
    const input =
        nn.input('input', {type: 'tensor-float32', dimensions: inputShape});
    const output = nn.transpose(input, permutation);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array(inputData));
    const outputBuffer = new Float32Array(24);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, expected);
  }

  it('transpose default', async function() {
    const inputShape = [2, 3, 4];
    const inputData = [
      0.43376675, 0.264609,   0.26321858, 0.04260185, 0.6862414,  0.26150206,
      0.04169406, 0.24857993, 0.14914423, 0.19905873, 0.33851373, 0.74131566,
      0.91501445, 0.21852633, 0.02267954, 0.22069663, 0.95799077, 0.17188412,
      0.09732241, 0.03296741, 0.04709655, 0.50648814, 0.13075736, 0.82511896
    ];
    const expected = [
      0.43376675, 0.91501445, 0.6862414,  0.95799077, 0.14914423, 0.04709655,
      0.264609,   0.21852633, 0.26150206, 0.17188412, 0.19905873, 0.50648814,
      0.26321858, 0.02267954, 0.04169406, 0.09732241, 0.33851373, 0.13075736,
      0.04260185, 0.22069663, 0.24857993, 0.03296741, 0.74131566, 0.82511896
    ];
    await checkTranspose(inputShape, inputData, expected);
  });

  it('transpose permutations', async function() {
    const permutations =
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    const inputShape = [2, 3, 4];
    const inputData = [
      0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
      0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
      0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
      0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487
    ];
    const expecteds = [
      [
        0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
        0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
        0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
        0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487
      ],
      [
        0.7760998,  0.8190919,  0.9306249,  0.8363521,  0.83241564, 0.00480607,
        0.10145967, 0.39479077, 0.39600816, 0.00533229, 0.5622921,  0.35415828,
        0.43689877, 0.4834097,  0.896649,   0.7603583,  0.6982117,  0.13060148,
        0.14368972, 0.7195266,  0.07824122, 0.11940759, 0.72893023, 0.33766487
      ],
      [
        0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.43689877, 0.7603583,
        0.14368972, 0.11940759, 0.8190919,  0.83241564, 0.39479077, 0.5622921,
        0.4834097,  0.6982117,  0.7195266,  0.72893023, 0.9306249,  0.00480607,
        0.39600816, 0.35415828, 0.896649,   0.13060148, 0.07824122, 0.33766487
      ],
      [
        0.7760998,  0.43689877, 0.8363521,  0.7603583,  0.10145967, 0.14368972,
        0.00533229, 0.11940759, 0.8190919,  0.4834097,  0.83241564, 0.6982117,
        0.39479077, 0.7195266,  0.5622921,  0.72893023, 0.9306249,  0.896649,
        0.00480607, 0.13060148, 0.39600816, 0.07824122, 0.35415828, 0.33766487
      ],
      [
        0.7760998,  0.8190919,  0.9306249,  0.43689877, 0.4834097,  0.896649,
        0.8363521,  0.83241564, 0.00480607, 0.7603583,  0.6982117,  0.13060148,
        0.10145967, 0.39479077, 0.39600816, 0.14368972, 0.7195266,  0.07824122,
        0.00533229, 0.5622921,  0.35415828, 0.11940759, 0.72893023, 0.33766487
      ],
      [
        0.7760998,  0.43689877, 0.8190919,  0.4834097,  0.9306249,
        0.896649,   0.8363521,  0.7603583,  0.83241564, 0.6982117,
        0.00480607, 0.13060148, 0.10145967, 0.14368972, 0.39479077,
        0.7195266,  0.39600816, 0.07824122, 0.00533229, 0.11940759,
        0.5622921,  0.72893023, 0.35415828, 0.33766487
      ]
    ];
    for (i in permutations) {
      await checkTranspose(
          inputShape, inputData, expecteds[i], permutations[i]);
    }
  });
});
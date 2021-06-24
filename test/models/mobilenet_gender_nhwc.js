'use strict';
import * as utils from '../utils.js';

/* eslint max-len: ["error", {"code": 120}] */

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = 'mobilenet_gender_nhwc';

describe('test mobilenet gender nhwc', function() {
  // It is based on Android NNAPI CTS: V1_0/mobilenet_224_gender_basic_fixed.mod.py
  // eslint-disable-next-line no-invalid-this
  this.timeout(0);
  let graph;
  let beforeNumBytes;
  let beforeNumTensors;
  before(async () => {
    if (typeof _tfengine !== 'undefined') {
      beforeNumBytes = _tfengine.memory().numBytes;
      beforeNumTensors = _tfengine.memory().numTensors;
    }
    const context = navigator.ml.createContext();
    const builder = new MLGraphBuilder(context);

    async function buildConv(input, weightsSubName, biasSubName, relu6, options) {
      const weightsName = `${testDataDir}/weights/op${weightsSubName}.npy`;
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = `${testDataDir}/weights/op${biasSubName}.npy`;
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      options.inputLayout = 'nhwc';
      const add = builder.add(
          builder.conv2d(input, weights, options),
          builder.reshape(bias, [1, 1, 1, -1]));
      // `relu6` in TFLite equals to `clamp` in WebNN API
      if (relu6) {
        return builder.clamp(
            add,
            {
              minValue: builder.constant(0.),
              maxValue: builder.constant(6.0),
            });
      }
      return add;
    }

    const strides = [2, 2];
    const autoPad = 'same-upper';
    const filterLayout = 'ohwi';
    const data = builder.input(
        'input', {type: 'float32', dimensions: [1, 224, 224, 3]});
    const conv0 = await buildConv(
        data, '2', '1', true, {strides, autoPad, filterLayout});
    const conv1 = await buildConv(conv0, '29', '28', true,
        {autoPad, groups: 16, filterLayout: 'ihwo'});
    const conv2 = await buildConv(
        conv1, '32', '31', true, {autoPad, filterLayout});
    const conv3 = await buildConv(conv2, '35', '34', true,
        {strides, autoPad, groups: 16, filterLayout: 'ihwo'});
    const conv4 = await buildConv(
        conv3, '38', '37', true, {autoPad, filterLayout});
    const conv5 = await buildConv(conv4, '41', '40', true,
        {autoPad, groups: 32, filterLayout: 'ihwo'});
    const conv6 = await buildConv(
        conv5, '44', '43', true, {autoPad, filterLayout});
    const conv7 = await buildConv(conv6, '47', '46', true,
        {strides, autoPad, groups: 32, filterLayout: 'ihwo'});
    const conv8 = await buildConv(
        conv7, '50', '49', true, {autoPad, filterLayout});
    const conv9 = await buildConv(conv8, '53', '52', true,
        {autoPad, groups: 64, filterLayout: 'ihwo'});
    const conv10 = await buildConv(
        conv9, '56', '55', true, {autoPad, filterLayout});
    const conv11 = await buildConv(conv10, '59', '58', true,
        {strides, autoPad, groups: 64, filterLayout: 'ihwo'});
    const conv12 = await buildConv(
        conv11, '62', '61', true, {autoPad, filterLayout});
    const conv13 = await buildConv(conv12, '65', '64', true,
        {autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv14 = await buildConv(
        conv13, '68', '67', true, {autoPad, filterLayout});
    const conv15 = await buildConv(conv14, '71', '70', true,
        {autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv16 = await buildConv(
        conv15, '74', '73', true, {autoPad, filterLayout});
    const conv17 = await buildConv(conv16, '77', '76', true,
        {autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv18 = await buildConv(
        conv17, '80', '79', true, {autoPad, filterLayout});
    const conv19 = await buildConv(conv18, '5', '4', true,
        {autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv20 = await buildConv(
        conv19, '8', '7', true, {autoPad, filterLayout});
    const conv21 = await buildConv(conv20, '11', '10', true,
        {autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv22 = await buildConv(
        conv21, '14', '13', true, {autoPad, filterLayout});
    const conv23 = await buildConv(conv22, '17', '16', true,
        {strides, autoPad, groups: 128, filterLayout: 'ihwo'});
    const conv24 = await buildConv(
        conv23, '20', '19', true, {autoPad, filterLayout});
    const conv25 = await buildConv(conv24, '23', '22', true,
        {autoPad, groups: 256, filterLayout: 'ihwo'});
    const conv26 = await buildConv(
        conv25, '26', '25', true, {autoPad, filterLayout});
    const averagePool2d = builder.averagePool2d(
        conv26, {strides, windowDimensions: [7, 7], layout: 'nhwc'});
    const conv27 = await buildConv(
        averagePool2d, '84', '83', false, {autoPad, filterLayout});
    const reshape = builder.reshape(conv27, [1, 11]);
    const softmax = builder.softmax(reshape);
    graph = await builder.build({softmax});
  });

  after(async () => {
    if (typeof _tfengine !== 'undefined') {
      // Check memory leaks.
      graph.dispose();
      const afterNumTensors = _tfengine.memory().numTensors;
      const afterNumBytes = _tfengine.memory().numBytes;
      assert(
          beforeNumTensors === afterNumTensors,
          `${afterNumTensors - beforeNumTensors} tensors are leaked.`);
      assert(
          beforeNumBytes === afterNumBytes,
          `${afterNumBytes - beforeNumBytes} bytes are leaked.`);
    }
  });

  async function testMobileNetGenderNhwc(inputFile, expectedFile) {
    const input = await utils.createTypedArrayFromNpy(new URL(inputFile, url));
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    const outputs = await graph.compute({'input': {data: input}});
    utils.checkShape(outputs.softmax.dimensions, [1, 11]);
    utils.checkValue(
        outputs.softmax.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  }

  it('test_data_set_0', async function() {
    await testMobileNetGenderNhwc(
        `${testDataDir}/test_data_set/0/op86.npy`,
        `${testDataDir}/test_data_set/0/op85.npy`);
  });
});

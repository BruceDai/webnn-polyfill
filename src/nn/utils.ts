import * as tf from '@tensorflow/tfjs-core';

import {OperandDescriptor} from './operand_descriptor';
import {OperandType} from './operand_type';
import {ArrayBufferView as TypedArray} from './types';

export function assert(expr: boolean, msg: string): void {
  if (!expr) {
    throw new Error(msg);
  }
}

export function isNumber(value: unknown): boolean {
  return typeof value === 'number';
}

export function isNumberArray(array: number[]): boolean {
  return array instanceof Array && array.every(v => isNumber(v));
}

export function isTensorType(type: OperandType): boolean {
  return type === 'tensor-float32' || type === 'tensor-float16' ||
      type === 'tensor-int32';
}

export function isTypedArray(array: TypedArray): boolean {
  return array instanceof Float32Array || array instanceof Int32Array ||
      array instanceof Uint32Array || array instanceof Int16Array ||
      array instanceof Uint16Array;
}

export function getTypedArray(type: OperandType): Float32ArrayConstructor|
    Int32ArrayConstructor|Uint32ArrayConstructor|Uint16ArrayConstructor {
  if (type === 'float32' || type === 'tensor-float32') {
    return Float32Array;
  } else if (type === 'int32' || type === 'tensor-int32') {
    return Int32Array;
  } else if (type === 'uint32') {
    return Uint32Array;
  } else if (type === 'float16' || type === 'tensor-float16') {
    return Uint16Array;
  } else {
    throw new Error('Type is not supported.');
  }
}

export function getDataType(type: OperandType): tf.DataType {
  if (type === 'float32' || type === 'tensor-float32') {
    return 'float32';
  } else if (type === 'int32' || type === 'tensor-int32') {
    return 'int32';
  } else {
    throw new Error('The operand type is not supported by TF.js.');
  }
}

export function createOperandDescriptorFromTensor(tensor: tf.Tensor):
    OperandDescriptor {
  let type: OperandType;
  if (tensor.dtype === 'float32') {
    if (tensor.rankType === tf.Rank.R0) {
      type = OperandType.float32;
    } else {
      type = OperandType['tensor-float32'];
    }
  } else if (tensor.dtype === 'int32') {
    if (tensor.rankType === tf.Rank.R0) {
      type = OperandType.int32;
    } else {
      type = OperandType['tensor-int32'];
    }
  }
  return {type, dimensions: tensor.shape} as OperandDescriptor;
}

export function validateOperandDescriptor(desc: OperandDescriptor): void {
  assert(desc.type in OperandType, 'The operand type is invalid.');
  if (isTensorType(desc.type)) {
    assert(
        isNumberArray(desc.dimensions), 'The operand dimensions is invalid.');
  } else {
    assert(
        desc.dimensions === undefined,
        'The operand dimensions is not required.');
  }
}

export function validateTypedArray(
    value: TypedArray, desc: OperandDescriptor): void {
  assert(isTypedArray(value), 'The value is not a typed array.');
  assert(
      value instanceof getTypedArray(desc.type),
      'The type of value is invalid.');
  if (!isTensorType(desc.type)) {
    assert(
        value.length === 1,
        `The value length ${value.length} is invalid, 1 is expected.`);
  } else {
    assert(
        value.length === sizeFromDimensions(desc.dimensions),
        `the value length ${value.length} is invalid, size of ` +
            `[${desc.dimensions}] ${sizeFromDimensions(desc.dimensions)} ` +
            'is expected.');
  }
}

export function createTensor(
    desc: OperandDescriptor, value: TypedArray|number): tf.Tensor {
  const dtype: tf.DataType = getDataType(desc.type);
  if (isTensorType(desc.type)) {
    validateTypedArray(value as TypedArray, desc);
    return tf.tensor(value as TypedArray, desc.dimensions, dtype);
  } else {
    if (typeof value === 'number') {
      return tf.scalar(value, dtype);
    } else {
      validateTypedArray(value, desc);
      return tf.scalar(value[0], dtype);
    }
  }
}

export function sizeFromDimensions(dim: number[]): number {
  if (typeof dim === 'undefined' || (isNumberArray(dim) && dim.length === 0)) {
    // scalar
    return 1;
  } else {
    return dim.reduce(
        (accumulator, currentValue) => accumulator * currentValue);
  }
}

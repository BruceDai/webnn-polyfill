import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
import {OperandType} from './operand_type';
import {ArrayBufferView as TypedArray} from './types';
import * as utils from './utils';

export class Constant extends Operand {
  readonly desc: OperandDescriptor;
  readonly value: number|TypedArray;

  static createScalar(value: number, type: OperandType = OperandType.float32):
      Constant {
    if (typeof type === 'undefined') {
      type = OperandType.float32;
    }
    utils.assert(type in OperandType, 'The operand type is invalid.');
    return new Constant({type} as OperandDescriptor, value);
  }

  static createTensor(desc: OperandDescriptor, value: TypedArray): Constant {
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(value, desc);
    return new Constant(desc, value);
  }

  private constructor(desc: OperandDescriptor, value: number|TypedArray) {
    super();
    this.desc = desc;
    this.value = value;
  }
}
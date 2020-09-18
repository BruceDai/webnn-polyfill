import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Transpose extends Operation {
  private permutation_: number[];

  constructor(input: Operand, permutation?: number[]) {
    super([input]);
    if (permutation) {
      utils.assert(
          utils.isNumberArray(permutation),
          'The permutation parameter is invalid.');
      this.permutation_ = permutation;
    } else {
      this.permutation_ = undefined;
    }
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.transpose(input, this.permutation_);
  }
}
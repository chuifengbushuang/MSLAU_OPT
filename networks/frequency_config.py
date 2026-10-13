"""Shared CLI contract for AFFE training and evaluation."""
import argparse
import math


def parse_enabled_stages(value):
    tokens = value.split(',')
    if len(tokens) != 4 or any(token.strip() not in {'0', '1'} for token in tokens):
        raise argparse.ArgumentTypeError('use four comma-separated 0/1 values, e.g. 1,1,1,1')
    return tuple(token.strip() == '1' for token in tokens)


def add_frequency_arguments(parser):
    parser.add_argument('--frequency_mode', choices=['none', 'naf2d'], default='none')
    parser.add_argument('--frequency_enabled_stages', type=parse_enabled_stages,
                        default=(True, True, True, True), metavar='1,1,1,1')
    parser.add_argument('--frequency_reg_weight', type=float, default=0.0)
    parser.add_argument('--frequency_band_edges', type=float, nargs=2, default=(1/3, 2/3))
    parser.add_argument('--frequency_band_weights', type=float, nargs=3, default=(1., 1., 1.))
    parser.add_argument('--frequency_alpha_init', type=float, default=0.1)


def frequency_model_kwargs(args):
    if not math.isfinite(args.frequency_reg_weight) or args.frequency_reg_weight < 0:
        raise ValueError('frequency_reg_weight must be finite and nonnegative')
    if args.frequency_mode != 'none' and args.decoder_mode != 'p5_sam2unet':
        raise ValueError('AFFE is only supported by p5_sam2unet')
    if args.frequency_reg_weight and args.frequency_mode != 'naf2d':
        raise ValueError('nonzero frequency_reg_weight requires naf2d')
    return {
        'frequency_mode': args.frequency_mode,
        'frequency_enabled_stages': args.frequency_enabled_stages,
        'frequency_band_edges': args.frequency_band_edges,
        'frequency_band_weights': args.frequency_band_weights,
        'frequency_alpha_init': args.frequency_alpha_init,
    }

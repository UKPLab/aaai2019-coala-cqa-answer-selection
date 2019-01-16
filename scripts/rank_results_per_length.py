import logging
from collections import OrderedDict

import click
import numpy as np

from experiment.qa.data.insuranceqa.reader.tsv_reader import TSVReader
from experiment.qa.data.insuranceqa.reader.v2_reader import V2Reader


def read_ranks(log):
    ranks = []
    start = False
    for l in log:
        if ' / test' in l:
            start = True
            ranks = []
        elif start:
            if 'Rank: ' in l:
                ranks.append(int(l.strip().split('Rank: ')[1]))
            else:
                start = False
    return ranks


@click.command()
@click.argument('dataset_path')
@click.argument('log', type=click.File('r'))
@click.option('--insuranceqa/--no-insuranceqa', default=False)
def main(dataset_path, log, insuranceqa):
    ranks = read_ranks(log)
    if insuranceqa:
        reader = V2Reader(dataset_path, lowercased=False, logger=logging.getLogger('root'))
    else:
        reader = TSVReader(dataset_path, lowercased=False, logger=logging.getLogger('root'), generated_questions_path=None)
    archive = reader.read()

    buckets = [50, 100, 150, 200, 250]
    ranks_per_bucket = OrderedDict([(b, []) for b in buckets])
    ranks_per_bucket['rest'] = []

    for qa, rank in zip(archive.test[0].qa, ranks):
        gt_len = np.mean([len(a.tokens) for a in qa.ground_truth])

        added = False
        for b in buckets:
            if gt_len <= b:
                ranks_per_bucket[b].append(rank)
                added = True
                break
        if not added:
            ranks_per_bucket['rest'].append(rank)

        # if b == 50:
        #     print(qa.question.text)
        #     print(qa.ground_truth[0].text)
        #     print('-'*5)

    for b, bucket_ranks in ranks_per_bucket.items():
        ranks_one = float(len([r for r in bucket_ranks if r == 1]))
        accuracy = ranks_one / len(bucket_ranks)
        print('Bucket {} (size={}) accuracy: {}'.format(b, len(bucket_ranks), accuracy))


if __name__ == '__main__':
    main()

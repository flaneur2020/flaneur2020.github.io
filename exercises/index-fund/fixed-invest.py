#!/usr/bin/env python
# coding: utf-8


"""
Usage:
        ./fixed-invest.py
"""

import os.path
import collections
from pprint import pprint


FundDataPoint = collections.namedtuple('FundDataPoint', ['date', 'price', 'cap', 'pe_ttm'])


def parse_csv(csv_path):
    points = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:-2]:
            cells = line.split(',')
            point = FundDataPoint(
                date=cells[0],
                price=float(cells[1]),
                cap=float(cells[2]),
                pe_ttm=float(cells[3]))
            points.append(point)
    return sorted(points, key=lambda p: p.date)


def calculate_fixed_invest_yield_rate(points, start_at, length, amount):
    invested_shares = 0
    invested_points = points[start_at:start_at+length]
    assert len(invested_points) == length
    for point in invested_points:
        invested_shares += amount / point.price
    last_price = points[-1].price
    return ((last_price * invested_shares) - (amount * length)) / (amount * length)


def fixed_invest_by_iterations(points, length, amount):
    # 迭代随机选择时间，根据投资时间长度计算预期收益
    stop_at = len(points) - length
    rows = []
    for start_at in xrange(0, stop_at):
        yield_rate = calculate_fixed_invest_yield_rate(points, start_at, length, amount)
        row = {
            'start_date': points[start_at].date,
            'end_date': points[start_at+length].date,
            'rate': round(yield_rate, 2)}
        rows.append(row)
    return rows


def main():
    csv_path = os.path.dirname(os.path.abspath(__file__)) + '/hs300_pe_ttm_20180401_171619.csv'
    points_by_month = parse_csv(csv_path)
    # calculate_fixed_invest_yield_rate(points_by_month, 0, 100, 1000)
    pprint(fixed_invest_by_iterations(points_by_month, 9, 1000))


if __name__ == '__main__':
    main()

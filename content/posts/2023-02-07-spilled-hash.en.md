---
date: "2023-02-07T00:00:00Z"
title: Spilled Hash Aggregation
language: "en"
---

I didn't pay much attention to Spilling before because an article about Presto mentioned that Facebook didn't enable it in production. Given Facebook's massive cluster size, memory can be considered infinite. The maintenance cost of adding extra disks to compute nodes might outweighs the benefits of Spilling.

However, recently I encountered some OOM cases. Without Facebook's large-scale permanent cluster size, Spilling is still a good option. Especially for products like DuckDB, "Never OOM" is indeed a big selling point. In resource-constrained environments, if battle-tested enough, it might have surprising effect in the ETL in a few years. Therefore, I want to understand how Spilling works.

Spilling is not a general capability of a Buffer Manager; it needs to be designed separately for different operators based on their characteristics. The main scenarios requiring Spilling are Hash Aggregation, Order By, and Hash Join. Let's first look at how Hash Aggregation handles Spilling.

## Hash Aggregation

Hash Aggregation involves running a query like `SELECT max(revenue) FROM daily_revenues GROUP BY year, city`. It uses `(year, city)` as the key for a hash table, pointing to an accumulator. If the cardinality of the GROUP BY key is too high, this hash table will consume too much memory and cause OOM.

So, we need to find a way to release the excessive memory usage.

The most straightforward way to release memory is similar to Linux's Swap. When memory is insufficient, swap memory to disk. When memory is sufficient, swap it back and continue processing.

But regardless of performance issues, is memory guaranteed to be sufficient when swapping back?

Given the nature of aggregation, this is easier to solve. There was a discussion in Datafusion:

> **alamb:** I think the current state of the art / best practice for dealing with grouping when the hashtable doesn't fit in memory is to:
> 1. Sort the data by groups and write that data to a spill file (ideally sharing / reusing the code used for externalized sort)
> 2. Create a new hash table, spilling repeatedly if needed
> 3. When all the input has been processed, then the spill files are read back and merged (again sharing code for externalized sort) and a merge group by is performed.

The gist is that if the memory usage of the Hash Table is too high during Aggregation, sort it and write it to disk.

Then, create a new Hash Table and continue the aggregation. When it's almost full, sort it and write it to disk again.

Finally, sort the hash table still in memory. Combine the in-memory hash table and multiple sorted files on disk, perform a multi-way merge, and stream the results of the multi-way merge. The memory required for multi-way merging is very low.

In this multi-way merge, not only is Merge Sort performed but aggregation is also continued. For example:

``` 
## spilled file 1:
(2020, beijing, $10)
(2020, new york, $20)

## in memory hash table
(2020, beijing, $1)
(2020, london, $23)
```

After merging:

``` 
(2020, beijing, $11)
(2020, new york, $20)
(2020, london, $23)
```

The difference from Swap is that the spilled memory does not need to be loaded back. Instead, by sorting, aggregation becomes a streaming operation of multi-way merging, consuming very little memory.

## Order By

Understanding how to handle Spilling in Hash Aggregation makes it easy to imagine how to handle Spilling in Order By. If memory is running low during sorting, the currently sorted part can be written to disk, and a new buffer can be allocated for sorting the rest until next spill. Finally, it would perform a multi-way merge and output the results in a stream.

## References

- [https://docs.google.com/document/d/16rm5VR1nGkY6DedMCh1NUmThwf3RduAweaBH9b1h6AY/edit](https://docs.google.com/document/d/16rm5VR1nGkY6DedMCh1NUmThwf3RduAweaBH9b1h6AY/edit)
- [https://github.com/duckdb/duckdb/pull/4970](https://github.com/duckdb/duckdb/pull/4970)
- [https://github.com/apache/arrow-datafusion/issues/1570](https://github.com/apache/arrow-datafusion/issues/1570)
- [https://deepai.org/publication/sort-based-grouping-and-aggregation](https://deepai.org/publication/sort-based-grouping-and-aggregation)
---
date: "2022-08-16T00:00:00Z"
title: Snapshot in Postgres
language: "en"
---

Recently, I wanted to learn about the postgres ecosystem. I didn't quite get its MVCC mechanism before, so I'm trying to understand it again. For now, I'll ignore the concurrency control and cleanup parts of MVCC and just focus on the Snapshot part.

## Tuple

Postgres doesn't have MySQL's UNDO log. Multi-version data (Tuples) are stored directly in the tablespace with version-related metadata. For now, let's just look at the xmin and xmax fields:

- xmin: Indicates the xid (transaction ID) when the Tuple was inserted.
- xmax: Indicates the xid when the Tuple was deleted.

For example, consider a Tuple that was inserted and committed:

```
| xmin | xmax | band  | fans  |
| 023  | 0    | tfboy | 9000w |
```

After deleting it in a new transaction:

```
| xmin | xmax | band  | fans  |
| 023  | 024  | tfboy | 9000w |
```

Here, xmax is set to the new transaction's xid.

What if you update this Tuple in a new transaction? Postgres treats updates as a delete + insert:

```
| xmin | xmax | band  | fans   |
| 023  | 024  | tfboy | 9000w  |
| 024  | 0    | tfboy | 10000w |
```

One counterintuitive thing is that whether the transaction COMMITs or ROLLBACKs, the Tuple in the tablespace doesn't change immediately. The transaction's commit status depends on the XACT structure.

XACT can be seen as a synonym for clog (Commit Log). It consists of a set of 8kb pages, with two bits for each transaction ID, indicating whether the transaction is In Progress, Committed, or Aborted. clog keeps appending and rotates every 256kb, but it doesn't grow indefinitely; the obsoleted clog files can be cleaned up during vacuum.

So when querying table data, you often need to binary search XACT (clog) to get the commit status of this row. Querying XACT has some overhead, so postgres also has two hint bits in the Tuple, indicating committed or rollbacked. If a Tuple is found to be committed/rollbacked during a read, a hint bit is set, so next time you don't need to access XACT again. It's kind of like a Read Repair process.

Why is it designed this way? Here's a quote from "MVCC in PostgreSQL-3. Row Versions":

> Why doesn't the transaction that performs the insert set these bits? When an insert is being performed, the transaction is unaware of whether it will be completed successfully. And at commit time, it's unclear which rows and in which pages were changed. There can be a lot of such pages, and it's impractical to keep track of them. Besides, some of the pages can be evicted to disk from the buffer cache; to read them again in order to change the bits would mean a considerable slowdown of the commit.

This makes XACT setting seem a bit like the Commit Point in percolator, where one atomic operation determines the commit status of N transaction participants.

## Snapshot

In storage layers like Rocksdb, which don't have uncommitted data, a Snapshot only needs a sequence number. But Postgres needs a bit more information:

- xmin: The earliest XID still active when the current transaction started. All data created before xmin should be visible (except for rollbacked data).
- xmax: The XID of the most recent committed transaction + 1. All data greater than xmax is invisible.
- xip[]: List of currently active transaction XIDs. Data related to active transactions should be invisible.

Refer to the diagram from "How Postgres Makes Transactions Atomic":

![](/images/2022-08-16-postgres-snapshot/Screen_Shot_2022-08-16_at_22.48.18.png)

In this Snapshot, transactions with 100 ≤ XID < 105 include 100 and 102. The data they committed is visible. Transactions with XID = 99 and 104 are not visible because they were rolled back. Transactions with XID = 101 and 103 are still in progress, so they are also not visible. It seems the visibility of transactions mainly depends on their status, while the xmin and xmax ranges help with pruning.

## References

- [1] [https://brandur.org/postgres-atomicity](https://brandur.org/postgres-atomicity#commit)
- [2] [https://habr.com/en/company/postgrespro/blog/477648/](https://habr.com/en/company/postgrespro/blog/477648/)
- [3] [https://philipmcclarence.com/what-is-the-pg_clog-and-the-clog/](https://philipmcclarence.com/what-is-the-pg_clog-and-the-clog/)

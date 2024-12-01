
fast append 是指每次追加一个 data file 都新生成一个 manifest 文件。需要再跑 REWRITE MANIFESTS 来整理这些 manifest 文件。
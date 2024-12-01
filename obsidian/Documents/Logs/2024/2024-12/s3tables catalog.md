### S3TablesCatalog

- newTableOps: 返回一个 S3TableOperations
- listNamespaces
- dropNamespace
- listTables
- dropTable

### S3TablesCatalogOperations.java

- doRefresh()
	- 调用 GetTableMetadataLocationRequest，得到一个 GetTableMetadataLocationResponse，返回一个 metadataLocation
	- 最后调用一下 refreshFromMetadataLocation
- doCommit()
	- this.tablesClient.getTableMetadataLocation() 拿一下 metadataLocation 中的 versionToken 信息；
	- this.tables.Client.updateTableMetadataLocation 更新一把，把 versionToken 带上，拿到一个返回的 versionToken；
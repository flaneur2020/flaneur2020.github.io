- Make endpoint attributes used by GIE components accessible via well defined Data Layer interfaces.

## Proposal

目前有两个 data source: 

1. 一个收集 pod 的 ip 和 label 的 pod reconciler；
2. 一个 metrics scraper，能够从每个 pod 去抓一组 metrics

这个 proposal 会引进两个新的接口，让它更有可扩展性：

1. 一个 attribute collection plugin 接口，负责从 data source 中捞取 attributes，并保存到 data layer；
2. 一个 Data source 插件，attribute collection 插件可以注册到里面

```go
// DataCollection interface consumes data updates from sources, stores
// it in the data layer for consumption.
// The plugin should not assume a deterministic invocation behavior beyond
// "the data layer believes the state should be updated"
type DataCollection interface {
    // Extract is called by data sources with (possibly) updated
    // data per endpoint. Extracted attributes are added to the 
    // Endpoint.
    Extract(ep Endpoint, data interface{}) error // or Collect?
}

// Endpoint interface allows setting and retrieving of attributes
// by a data collector.
// Note that actual endpoint structure would be something like (pseudocode)
// type EndpointState struct {
//   address
//   ...
//   data map[string]interface{}
// }
// The plugin interface would only mutate the `data` map
type Endpoint interface {
   // StoreAttributes sets the data for the Endpoint on behalf
   // of the named collection Plugin
   StoreAttributes(collector string, data interface{}) error
   
   // GetAttributes retrieves the attributes of the named collection
   // plugin for the Endpoint
   GetAttributes(collector string) (interface{}, error)
}

// DataLayerSourcesRegistry include the list of available 
// Data Sources (interface defined below) in the system.
// It is accompanied by functions (not shown) to register
// and retrieve sources
type DataLayerSourcesRegistry map[string]DataSource 

// DataSource interface represents a data source that tracks
// pods/resources and notifies data collection plugins to
// extract relevant attributes.
type DataSource interface {
    // Type of data available from this source
    Type() string

    // Start begins the data collection and notification loop
    Start(ctx context) error

    // Stop terminates data collection
    Stop() error

    // Subscribe a collector to receive updates for tracked endpoints
    Subscribe(collector DataCollection) error

    // UpdateEndpoints replaces the set of pods/resources tracked by
    // this source.
    // Alternative: add/remove individual endpoints?
    UpdateEndpoints(epIDs []string) error 
}
```

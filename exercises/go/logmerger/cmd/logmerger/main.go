package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/fleurer/logmerger/logmerger"
)

type LogMergerOptions struct {
	Consumers []struct {
		Name string `json:"name"`
		Kind string `json:"kind"`
	} `json:"consumers"`

	Throttling struct {
		Enabled      bool   `json:"enabled"`
		Threshold    uint64 `json:"threshold"`
		TimeWindowMs int64  `json:"timeWindowMs"`
	} `json:"throttling"`

	Stats struct {
		Enabled bool   `json:"enabled"`
		Addr    string `json:"addr"`
	} `json:"stats"`
}

func (opt *LogMergerOptions) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(file)
	return dec.Decode(opt)
}

type LogMerger struct {
	opts      *LogMergerOptions
	consumers map[string]logmerger.LogConsumer

	closed bool
	wg     sync.WaitGroup
	quitc  chan struct{}
}

func setupLogMerger(opts *LogMergerOptions) *LogMerger {
	// setup consumers
	consumers := map[string]logmerger.LogConsumer{}
	for _, consumerConfig := range opts.Consumers {
		consumer := logmerger.NewFakeLogConsumer()
		consumers[consumerConfig.Name] = consumer
	}

	return &LogMerger{
		consumers: consumers,
		opts:      opts,
		quitc:     make(chan struct{}),
		wg:        sync.WaitGroup{},
	}
}

func (lm *LogMerger) Tailf() {
	var it logmerger.LogIterator

	lm.wg.Add(1)
	defer lm.wg.Done()

	// setup merge iterator, this might block
	its := []logmerger.LogIterator{}
	for _, c := range lm.consumers {
		its = append(its, c.Iterator())
	}
	it = logmerger.NewMergedIterator(its)

	// setup throttling if nesscary
	if lm.opts.Throttling.Enabled {
		timeWindow := time.Duration(lm.opts.Throttling.TimeWindowMs) * time.Millisecond
		it = logmerger.NewThrottledIterator(it, lm.opts.Throttling.Threshold, timeWindow)
	}

	// start tailing
	for ; it.Current() != nil; it.Next() {
		select {
		case <-lm.quitc:
			return
		default:
		}

		lg := it.Current()
		log.Printf("LOG: %#v", lg)
	}
}

func (lm *LogMerger) Close() {
	if lm.closed {
		return
	}
	close(lm.quitc)
	lm.closed = true
	lm.wg.Wait()
}

func (lm *LogMerger) ServeHttp(addr string) {
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		stats := map[string]logmerger.LogConsumerStats{}
		for name, c := range lm.consumers {
			stats[name] = c.Stats()
		}
		enc := json.NewEncoder(w)
		enc.Encode(stats)
	})

	log.Printf("listening %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Printf("listen failed: %s", err)
	}
}

func main() {
	var (
		configPath string
		position   int64
	)

	flag.StringVar(&configPath, "c", "", "config path")
	flag.Int64Var(&position, "p", -1, "start position")
	flag.Parse()

	opts := &LogMergerOptions{}
	if err := opts.Load(configPath); err != nil {
		log.Fatalf("failed to load config")
		return
	}

	var wg sync.WaitGroup
	quitConsumersc := make(chan struct{})

	lm := setupLogMerger(opts)
	for name, consumer := range lm.consumers {
		go func(name string, c logmerger.LogConsumer) {
			wg.Add(1)
			log.Printf("consumer#%s start", name)
			c.Run(position, quitConsumersc)
			c.Close()
			wg.Done()
			log.Printf("consumer#%s closed", name)
		}(name, consumer)
	}

	go func() {
		wg.Add(1)
		lm.Tailf()
		wg.Done()
	}()

	if opts.Stats.Enabled {
		go lm.ServeHttp(opts.Stats.Addr)
	}

	sc := make(chan os.Signal, 1)
	signal.Notify(sc, os.Kill, os.Interrupt, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM)
	<-sc

	// 确保先关闭 LogMerger, 再关闭 Consumers
	log.Printf("closing...")
	lm.Close()
	close(quitConsumersc)
	wg.Wait()
	log.Printf("closed")
}

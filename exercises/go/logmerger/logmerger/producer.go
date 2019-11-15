package logmerger

import (
	"context"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

var (
	token int64
	step  int64

	maxSleepInterval int64 = 5
	maxGap           int64 = 10
)

type FakeProducer struct {
	logc chan Log
}

func (p *FakeProducer) Output() chan Log {
	return p.logc
}

func (p *FakeProducer) Produce(ctx context.Context) {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		p.generateDatas(ctx)
	}()
	go func() {
		defer wg.Done()
		p.generateDatas(ctx)
	}()
	wg.Wait()
}

func (p *FakeProducer) generateDatas(ctx context.Context) {
	for {
		prepare := incrementToken()
		sleep(maxSleepInterval)

		select {
		case <-ctx.Done():
			return
		case p.logc <- Log{kind: "prepare", prepareToken: prepare}:
		}
		sleep(maxSleepInterval)

		commit := incrementToken()
		sleep(maxSleepInterval)

		select {
		case <-ctx.Done():
			return
		case p.logc <- Log{kind: "commit", prepareToken: prepare, commitToken: commit}:
		}
		sleep(10 * maxSleepInterval)
	}
}

func incrementToken() int64 {
	return atomic.AddInt64(&token, rand.Int63()%maxGap+1)
}

func sleep(factor int64) {
	interval := atomic.AddInt64(&step, 3)%factor + 1
	waitTime := time.Duration(rand.Int63() % interval)
	time.Sleep(waitTime * time.Millisecond)
}

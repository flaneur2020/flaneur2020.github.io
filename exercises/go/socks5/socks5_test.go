package socks5

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_handshake(t *testing.T) {
	conn, err := Dial("127.0.0.1:8081", SocksAddrTypeDomain, []byte("twitter.com"), 443)
	assert.Nil(t, err)
	assert.NotNil(t, conn)
}

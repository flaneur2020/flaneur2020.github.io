package socks5

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"time"
)

type Socks5Addr struct {
	addrType byte
	addr     []byte
	port     uint16
}

type SocksConn struct {
	conn net.Conn
}

var (
	errHandshake           = errors.New("socks handshake failed")
	errConnConfirmationErr = errors.New("socks conn confirmation error")
	errAuthFailed          = errors.New("socks auth failed")
)

const (
	SocksVer5                 = 5
	SocksNoAuth               = 0
	SocksCmdConnect           = 1
	SocksStatusRequestGranted = 0
	SocksMaxAddrLen           = 1 + 1 + 255 + 2
	SocksAddrTypeIPv4         = 1
	SocksAddrTypeDomain       = 3
	SocksAddrTypeIPv6         = 4
)

var _ net.Conn = &SocksConn{}

func Dial(proxyAddr string, addrType byte, addr []byte, port uint16) (*SocksConn, error) {
	conn, err := net.Dial("tcp", proxyAddr)
	if err != nil {
		return nil, err
	}

	c := &SocksConn{
		conn: conn,
	}

	if err := c.handshake(); err != nil {
		return nil, err
	}

	socks5Addr := &Socks5Addr{addrType: addrType, addr: addr, port: port}
	if err := c.writeConnectRequest(socks5Addr); err != nil {
		return nil, err
	}

	if err := c.readConnectResponse(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *SocksConn) Read(b []byte) (int, error) {
	return c.conn.Read(b)
}

func (c *SocksConn) Write(buf []byte) (int, error) {
	return c.conn.Write(buf)
}

func (c *SocksConn) Close() error {
	return c.conn.Close()
}
func (c *SocksConn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

func (c *SocksConn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

func (c *SocksConn) SetDeadline(t time.Time) error {
	return c.conn.SetDeadline(t)
}

func (c *SocksConn) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

func (c *SocksConn) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}

func (c *SocksConn) handshake() error {
	if err := c.writeAuthRequest(); err != nil {
		return err
	}

	if err := c.readAuthResponse(); err != nil {
		return err
	}

	return nil
}

func (c *SocksConn) writeAuthRequest() error {
	buf := []byte{SocksVer5, 1, SocksNoAuth}
	_, err := c.conn.Write(buf)
	return err
}

func (c *SocksConn) readAuthResponse() error {
	var buf [2]byte
	_, err := io.ReadFull(c.conn, buf[:])
	if err != nil {
		return err
	}
	expectedBuf := []byte{SocksVer5, 0}
	if !bytes.Equal(buf[:], expectedBuf) {
		return errAuthFailed
	}
	return nil
}

func (c *SocksConn) writeConnectRequest(addr *Socks5Addr) error {
	headerBuf := []byte{SocksVer5, SocksCmdConnect, 0}
	if _, err := c.conn.Write(headerBuf); err != nil {
		return err
	}

	if err := c.writeSocks5Addr(addr); err != nil {
		return err
	}

	return nil
}

func (c *SocksConn) readConnectResponse() error {
	var headBuf [3]byte
	_, err := io.ReadFull(c.conn, headBuf[:])
	if err != nil {
		return fmt.Errorf("read response head failed: %w", err)
	}

	status := headBuf[1]
	if status != SocksStatusRequestGranted {
		return fmt.Errorf("unexpected status: %d", status)
	}

	socks5Addr := Socks5Addr{}
	if err := c.readSocks5Addr(&socks5Addr); err != nil {
		return err
	}

	return nil
}

func (c *SocksConn) writeSocks5Addr(addr *Socks5Addr) error {
	buf := []byte{}
	buf = append(buf, addr.addrType)
	switch addr.addrType {
	case SocksAddrTypeIPv4:
		buf = append(buf, addr.addr[:net.IPv4len]...)
	case SocksAddrTypeIPv6:
		buf = append(buf, addr.addr[:net.IPv6len]...)
	case SocksAddrTypeDomain:
		buf = append(buf, byte(len(addr.addr)))
		buf = append(buf, addr.addr...)
	}

	var portBuf [2]byte
	binary.BigEndian.PutUint16(portBuf[:], addr.port)
	buf = append(buf, portBuf[:]...)

	_, err := c.conn.Write(buf)
	return err
}

func (c *SocksConn) readSocks5Addr(socks5Addr *Socks5Addr) error {
	var (
		addrTypeBuf []byte = []byte{0}
		addrSizeBuf []byte = []byte{0}
		addrBuf     []byte
		portBuf     []byte = []byte{0, 0}
	)

	if _, err := io.ReadFull(c.conn, addrTypeBuf); err != nil {
		return err
	}

	switch addrTypeBuf[0] {
	case SocksAddrTypeIPv4:
		addrBuf = make([]byte, net.IPv4len)
	case SocksAddrTypeIPv6:
		addrBuf = make([]byte, net.IPv6len)
	case SocksAddrTypeDomain:
		_, err := io.ReadFull(c.conn, addrSizeBuf)
		if err != nil {
			return err
		}
		addrBuf = make([]byte, addrSizeBuf[0])
	}

	if _, err := io.ReadFull(c.conn, addrBuf); err != nil {
		return err
	}

	if _, err := io.ReadFull(c.conn, portBuf); err != nil {
		return err
	}

	socks5Addr.addrType = addrTypeBuf[0]
	socks5Addr.addr = addrBuf
	socks5Addr.port = binary.BigEndian.Uint16(portBuf)
	return nil
}

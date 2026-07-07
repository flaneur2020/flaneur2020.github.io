// Command vanillafuse mounts a synthetic read-only FUSE filesystem that
// exposes three fixed files (file1, file2, file3), each 10 GiB in size,
// where every byte reads as 'x'. Nothing is stored on disk; reads are
// synthesised in memory. The intended use is to benchmark the throughput
// of the FUSE read path itself, independent of any backing store.
//
// Usage:
//
//	vanillafuse --mount /mnt/vanillafuse
//	vanillafuse --mount /mnt/vanillafuse --allow-other
//
// Press Ctrl-C (or send SIGTERM) to unmount and exit.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hanwen/go-fuse/v2/fs"
	gofuse "github.com/hanwen/go-fuse/v2/fuse"
)

const (
	// fileSize is the synthetic size of every regular file: 10 GiB.
	fileSize int64 = 10 << 30
	// fillByte is the single byte returned at every offset in every file.
	fillByte byte = 'x'
)

// attrTimeout is the TTL advertised to the kernel for cached attributes
// and directory entries. The layout is immutable for the lifetime of the
// mount, so we use a long TTL and let the kernel cache aggressively.
// Declared as a var (not const) so its address can be taken for the
// fs.Options pointer fields below.
var attrTimeout = time.Hour

// fileNames are the entries served at the mount root, in this order.
// Inode numbers: root=1, file1=2, file2=3, file3=4.
var fileNames = []string{"file1", "file2", "file3"}

// xfill is a 1 MiB buffer of fillByte, used to fill read destinations
// via copy() (memcpy). Sized to stay L2-resident while keeping the
// per-read loop iteration count low; this is faster on large reads
// than a byte-by-byte fill or bytes.Repeat (which would allocate).
var xfill [1 << 20]byte

func init() {
	for i := range xfill {
		xfill[i] = fillByte
	}
}

// fillWithX fills b with fillByte using the pre-allocated xfill buffer.
func fillWithX(b []byte) {
	for off := 0; off < len(b); off += len(xfill) {
		end := off + len(xfill)
		if end > len(b) {
			end = len(b)
		}
		copy(b[off:end], xfill[:end-off])
	}
}

// inoFor returns the stable inode for a file name. Root=1, files=2..N+1.
// Returns 0 for unknown names so callers can use it as a presence check.
func inoFor(name string) uint64 {
	for i, fn := range fileNames {
		if fn == name {
			return uint64(2 + i)
		}
	}
	return 0
}

// stableMode returns the syscall mode bits (S_IFMT | perms). Directories
// are 0555, files are 0444: both read-only.
func stableMode(isDir bool) uint32 {
	if isDir {
		return uint32(syscall.S_IFDIR | 0555)
	}
	return uint32(syscall.S_IFREG | 0444)
}

// vanillaNode is the go-fuse node for both the root directory and the
// synthetic files. isDir distinguishes them; the same type implements
// every read-only FUSE callback we need.
type vanillaNode struct {
	fs.Inode
	name  string
	isDir bool
	ino   uint64
}

// Compile-time assertions that vanillaNode implements the go-fuse
// interfaces we rely on.
var (
	_ fs.InodeEmbedder = (*vanillaNode)(nil)
	_ fs.NodeGetattrer = (*vanillaNode)(nil)
	_ fs.NodeLookuper  = (*vanillaNode)(nil)
	_ fs.NodeReaddirer = (*vanillaNode)(nil)
	_ fs.NodeOpener    = (*vanillaNode)(nil)
	_ fs.NodeReader    = (*vanillaNode)(nil)
	_ fs.NodeStatfser  = (*vanillaNode)(nil)
)

func (n *vanillaNode) String() string {
	if n.isDir {
		return "/"
	}
	return n.name
}

// Getattr fills attribute information from the in-memory node.
func (n *vanillaNode) Getattr(ctx context.Context, fh fs.FileHandle, out *gofuse.AttrOut) syscall.Errno {
	out.Ino = n.ino
	out.Mode = stableMode(n.isDir)
	out.Nlink = 1
	out.Uid = 0
	out.Gid = 0
	out.Rdev = 0
	out.Blksize = 4096
	if !n.isDir {
		out.Size = uint64(fileSize)
		out.Blocks = uint64((fileSize + 511) / 512)
	}
	out.SetTimeout(attrTimeout)
	return 0
}

// Lookup resolves a single child name in a directory node. Returning
// ENOENT caches the negative lookup for the entry timeout as well.
func (n *vanillaNode) Lookup(ctx context.Context, name string, out *gofuse.EntryOut) (*fs.Inode, syscall.Errno) {
	if !n.isDir {
		return nil, syscall.ENOTDIR
	}
	ino := inoFor(name)
	if ino == 0 {
		return nil, syscall.ENOENT
	}
	child := &vanillaNode{name: name, isDir: false, ino: ino}
	out.Ino = ino
	out.Mode = stableMode(false)
	out.Nlink = 1
	out.Uid = 0
	out.Gid = 0
	out.Rdev = 0
	out.Blksize = 4096
	out.Size = uint64(fileSize)
	out.Blocks = uint64((fileSize + 511) / 512)
	out.SetEntryTimeout(attrTimeout)
	out.SetAttrTimeout(attrTimeout)
	stable := fs.StableAttr{Ino: ino, Mode: stableMode(false)}
	return n.NewInode(ctx, child, stable), 0
}

// Readdir lists the direct children of the root directory. "." and ".."
// are synthesised by go-fuse and must not be included.
func (n *vanillaNode) Readdir(ctx context.Context) (fs.DirStream, syscall.Errno) {
	if !n.isDir {
		return nil, syscall.ENOTDIR
	}
	entries := make([]gofuse.DirEntry, 0, len(fileNames))
	for _, fn := range fileNames {
		entries = append(entries, gofuse.DirEntry{
			Ino:  inoFor(fn),
			Name: fn,
			Mode: stableMode(false),
		})
	}
	return fs.NewListDirStream(entries), 0
}

// Open rejects write-related flags and refuses to open directories. The
// returned FileHandle is nil: reads are stateless and dispatched on the
// node itself via NodeReader.
func (n *vanillaNode) Open(ctx context.Context, flags uint32) (fs.FileHandle, uint32, syscall.Errno) {
	const writeMask = uint32(syscall.O_WRONLY | syscall.O_RDWR | syscall.O_CREAT |
		syscall.O_TRUNC | syscall.O_APPEND)
	if flags&writeMask != 0 {
		return nil, 0, syscall.EROFS
	}
	if n.isDir {
		return nil, 0, syscall.EISDIR
	}
	return nil, 0, 0
}

// Read synthesises fillByte for [off, off+len(dest)) clamped to fileSize.
// No I/O is performed; the destination is filled in memory and handed
// back to the framework without further copying. EOF (off >= fileSize)
// and zero-length reads return an empty ReadResult.
func (n *vanillaNode) Read(ctx context.Context, f fs.FileHandle, dest []byte, off int64) (gofuse.ReadResult, syscall.Errno) {
	_ = f // stateless: no file handle is used
	if len(dest) == 0 || off >= fileSize {
		return gofuse.ReadResultData(nil), 0
	}
	end := off + int64(len(dest))
	if end > fileSize {
		end = fileSize
	}
	nbytes := int(end - off)
	fillWithX(dest[:nbytes])
	return gofuse.ReadResultData(dest[:nbytes]), 0
}

// Statfs reports a small synthetic filesystem with no free space.
func (n *vanillaNode) Statfs(ctx context.Context, out *gofuse.StatfsOut) syscall.Errno {
	const bsize = uint32(4096)
	blocksPerFile := uint64((fileSize + int64(bsize) - 1) / int64(bsize))
	out.Blocks = blocksPerFile * uint64(len(fileNames))
	out.Bfree = 0
	out.Bavail = 0
	out.Files = uint64(len(fileNames))
	out.Ffree = 0
	out.Bsize = bsize
	out.NameLen = 255
	out.Frsize = bsize
	return 0
}

func main() {
	mountPoint := flag.String("mount", "", "Mount point directory (required). May also be supplied positionally.")
	allowOther := flag.Bool("allow-other", false, "Allow non-root users to access the mount (sets FUSE allow_other).")
	flag.Parse()

	if *mountPoint == "" && flag.NArg() >= 1 {
		*mountPoint = flag.Arg(0)
	}
	if *mountPoint == "" {
		log.Fatal("vanillafuse: --mount is required")
	}

	st, err := os.Stat(*mountPoint)
	if err != nil {
		log.Fatalf("vanillafuse: stat mount point %q: %v", *mountPoint, err)
	}
	if !st.IsDir() {
		log.Fatalf("vanillafuse: mount point %q is not a directory", *mountPoint)
	}

	root := &vanillaNode{name: "", isDir: true, ino: 1}

	opts := &fs.Options{
		EntryTimeout:    &attrTimeout,
		AttrTimeout:     &attrTimeout,
		NegativeTimeout: &attrTimeout,
		MountOptions: gofuse.MountOptions{
			AllowOther:    *allowOther,
			Options:       []string{"ro"},
			FsName:        "vanillafuse",
			Name:          "vanillafuse",
			DisableXAttrs: true,
		},
	}

	server, err := fs.Mount(*mountPoint, root, opts)
	if err != nil {
		log.Fatalf("vanillafuse: mount %q: %v", *mountPoint, err)
	}
	log.Printf("vanillafuse: mounted at %s; %d files of %d bytes each, every byte = %q (Ctrl-C to unmount)",
		*mountPoint, len(fileNames), fileSize, fillByte)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("vanillafuse: received %v, unmounting...", sig)
		if err := server.Unmount(); err != nil {
			log.Printf("vanillafuse: unmount failed (%v); forcing exit", err)
			os.Exit(1)
		}
	}()

	server.Wait()
	log.Printf("vanillafuse: unmounted %s, bye", *mountPoint)
}

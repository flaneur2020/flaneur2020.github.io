#!/usr/bin/env python
# coding: utf-8

"""sync-notion.py

Usage:
  sync-notion.py list
  sync-notion.py sync <POST>

Options:
  -h --help
"""

import os
import requests
import sys
import docopt
import typing
import notion.block
import os.path
from urllib.parse import urlparse
from itertools import takewhile
from notion.client import NotionClient

# https://raw.githubusercontent.com/echo724/notion2md/main/notion2md/exporter.py


class PageExporter:
    def __init__(
        self, page: notion.block.PageBlock, images_dir: str, images_base_url: str
    ):
        self._page = page
        self._images_dir = images_dir
        self._images_base_url = images_base_url

    def export_markdown(self, meta={}):
        md = "---\n"
        md += "title: %s\n" % self._page.title
        for k, v in meta.items():
            md += "%s: %s\n" % (k, v)
        md += "---\n\n"

        blocks = self._convert_children_to_blocks(self._page.children)
        images_map = self._download_images(blocks)
        md += self._blocks2md(blocks, 0, images_map).strip()
        return md

    def _convert_children_to_blocks(
        self, children: notion.block.Children
    ) -> typing.List[notion.block.Block]:
        blocks = typing.cast(typing.List[notion.block.Block], list(children))
        return blocks

    def _flatten_blocks_with_children(
        self,
        blocks: typing.List[notion.block.Block],
        indent: int = 0,
    ) -> typing.List[typing.Tuple[notion.block.Block, int]]:
        result = []
        for block in blocks:
            result.append((block, indent))
            if block.children and len(block.children) > 0:
                children_blocks = self._convert_children_to_blocks(block.children)
                result.extend(
                    self._flatten_blocks_with_children(children_blocks, indent + 1)
                )
        return result

    def _download_images(self, blocks: typing.List[notion.block.Block]) -> typing.Dict[str, str]:
        images_map = {}
        block_with_indents = self._flatten_blocks_with_children(blocks)
        for block, _ in block_with_indents:
            if block.type == "image":
                block = typing.cast(notion.block.ImageBlock, block)
                _, image_url = download_image(
                    block.source, self._images_dir, self._images_base_url
                )
                filename = parse_image_filename(block.source)
                images_map[filename] = image_url
        return images_map

    def _blocks2md(
        self,
        blocks: typing.List[notion.block.Block],
        indent: int = 0,
        images_map: typing.Dict[str, str] = {},
    ):
        i = 0
        md = ""
        list_btypes = [
            notion.block.BulletedListBlock,
            notion.block.NumberedListBlock,
            notion.block.TodoBlock,
        ]
        block_with_indents = self._flatten_blocks_with_children(blocks)
        while i < len(block_with_indents):
            block, indent = block_with_indents[i]
            if type(block) in list_btypes:
                list_items = list(
                    takewhile(
                        lambda x: type(x[0]) in list_btypes, block_with_indents[i:]
                    )
                )
                for block, indent in list_items:
                    md += self._block2md(block, indent, images_map)
                    md += "\n"
                md += "\n"
                i += len(list_items)
            else:
                md += self._block2md(block, indent, images_map)
                md += "\n\n"
                i += 1
        return md

    def _block2md(self, block, indent, images_map):
        md = ""
        btype = block.type
        if btype == "header":
            md += "# " + filter_inline_math(block)
        elif btype == "sub_header":
            md += "## " + filter_inline_math(block)
        elif btype == "sub_sub_header":
            md += "### " + filter_inline_math(block)
        elif btype == "text":
            md += filter_inline_math(block)
        elif btype == "bookmark":
            md += format_link(block.title, block.link)
        elif (
            btype == "video"
            or btype == "file"
            or btype == "audio"
            or btype == "pdf"
            or btype == "gist"
        ):
            md += format_link(block.source, block.source)
        elif btype == "bulleted_list" or btype == "toggle":
            md += "- " + filter_inline_math(block)
        elif btype == "numbered_list":
            md += "1. " + filter_inline_math(block)
        elif btype == "code":
            md += "``` " + block.language.lower() + "\n" + block.title + "\n```"
        elif btype == "equation":
            md += "$$" + block.latex + "$$"
        elif btype == "divider":
            md += "---"
        elif btype == "to_do":
            if block.checked:
                md += "- [x] " + block.title
            else:
                md += "- [ ]" + block.title
        elif btype == "quote":
            md += "> " + block.title
        elif btype == "column" or btype == "column_list":
            md += ""
        elif btype == "table_of_contents":
            # 忽略 toc
            pass
        elif btype == "image":
            image_filename = parse_image_filename(block.source)
            image_url = images_map.get(image_filename, block.source)
            md += "![](%s)" % image_url
        else:
            raise Exception("unsupport block type: %s" % btype)

        indent_spaces = "    " * indent
        return indent_spaces + md


def format_link(name, url):
    """make markdown link format string"""
    return "[" + name + "]" + "(" + url + ")"


def table_to_markdown(table):
    md = ""
    md += join_with_vertical(table[0])
    md += "\n---|---|---\n"
    for row in table[1:]:
        if row != table[1]:
            md += "\n"
        md += join_with_vertical(row)
    return md


def join_with_vertical(list):
    return " | ".join(list)


def filter_inline_math(block):
    """This function will get inline math code and append it to the text"""
    text = ""
    elements = block.get("properties", {}).get("title", [])
    for i in elements:
        if i[0] == "⁍":
            text += "$$" + i[1][0][1] + "$$"
        else:
            text += block.title
    return text


def filter_source_url(block):
    try:
        return block.get("properties")["source"][0][0]
    except:
        return block.title


def parse_image_filename(source_url: str) -> str:
    url_path = urlparse(source_url).path
    filename = url_path.split("/")[-1]
    return filename


def download_image(source_url, images_dir, images_base_url):
    filename = parse_image_filename(source_url)
    download_path = os.path.join(images_dir, filename)
    image_url = os.path.join(images_base_url, filename)
    if os.path.exists(download_path):
        print("existed image %s" % download_path)
        return False, image_url
    print("downloading image %s" % download_path)
    resp = requests.get(source_url, allow_redirects=True)
    os.makedirs(images_dir, exist_ok=True)
    with open(download_path, "w+") as f:
        f.write(str(resp.content))
    return True, image_url


POST_URLS = {
    "2021-08-01-badger-txn": "https://www.notion.so/fleuria/badger-bdbd1620efd84038afedd9efc708ee66",
    "2021-08-14-rocksdb-txn": "https://www.notion.so/fleuria/rocksdb-a1bd4ae158be4b77b37e75bb210e105f",
    "2021-09-06-crdb-txn": "https://www.notion.so/fleuria/crdb-b54fc0cff91244a9acebd87d4ea6329c",
}


def main():
    opts = docopt.docopt(str(__doc__))
    if opts["list"]:
        items = POST_URLS.items()
        items = sorted(items, key=lambda x: x[0])
        for k, v in items:
            print("%s -- %s" % (k, v))
    elif opts["sync"]:
        post_name = opts["<POST>"]
        url = POST_URLS.get(post_name)
        if not url:
            print("%s not found" % post_name)
            sys.exit(1)
        download_post(post_name, url)


def download_post(post_name, url):
    token_v2 = open(".notion-token").read().strip()
    client = NotionClient(token_v2=token_v2)
    page = typing.cast(notion.block.PageBlock, client.get_block(url))
    images_dir = "images/%s/" % post_name
    exporter = PageExporter(page, images_dir, "/" + images_dir)
    md = exporter.export_markdown({"layout": "post"})
    file_name = "_posts/%s.md" % post_name
    with open(file_name, "w+") as f:
        f.write(md)


if __name__ == "__main__":
    main()

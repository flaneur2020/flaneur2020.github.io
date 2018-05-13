# coding: utf-8
import json
import requests


class AnnotationDTO(object):
    def __init__(self, data):
        self.book_title = data['book']['title']
        self.book_url = data['book']['url']
        self.chapter = data['chapter']
        self.content = data['content']
        self.time = data['time']
        self.date = data['time'].split(' ')[0]

    def render_content_as_markdown(self):
        output = ""
        content = {}
        raw_content = self.content
        if not raw_content.startswith(u'{'):
            return self._render_old_style_content_as_markdown(self.content)
        content = json.loads(self.content.encode('utf-8'))
        for block in content.get('blocks', []):
            if block['type'] == 'blockquote':
                output += u'> %s\n>\n' % block['text']
            else:
                output += u'\n%s\n\n' % block['text']
        return output

    def _render_old_style_content_as_markdown(self, content):
        return content

    def __repr__(self):
        return u'AnnotationDTO<%s>' % self.__dict__


class AnnotationExporter(object):
    def __init__(self):
        pass

    def export(self, user_name, dir_path='./notes'):
        annotations = self._fetch_all(user_name)
        book_annotations_pairs = self._collect_annotations_by_book(annotations)
        for book_title, annotations in book_annotations_pairs:
            last_date = annotations[0].date
            gen_path = u"%s/%s.md" % (dir_path, self._cook_target_path(book_title, last_date))
            with open(gen_path, 'w+') as f:
                print u"render %s" % gen_path
                md_content = self._render_annotations_of_one_book(annotations)
                f.write(md_content.encode('utf-8'))
        gen_path = '%s/index.md' % dir_path
        with open(gen_path, 'w+') as f:
            print u"render %s" % gen_path
            md_content = self._render_annotations_index(book_annotations_pairs)
            f.write(md_content.encode('utf-8'))

    def _render_annotations_of_one_book(self, annotations):
        output = u''
        output += u'---\n'
        output += u'layout: default\n'
        output += u'title: %s\n' % annotations[0].book_title
        output += u'---\n\n'
        output += u'# 读书笔记: %s\n\n' % annotations[0].book_title
        for annotation in annotations:
            output += u'\n## %s\n\n' % annotation.chapter
            output += annotation.render_content_as_markdown()
        return output

    def _render_annotations_index(self, book_annotations_pairs):
        output = u'---\n'
        output += u'layout: default\n'
        output += u'title: Book Notes\n'
        output += u'---\n\n'
        output += u'# Book Notes\n\n'
        for book_title, annotations in book_annotations_pairs:
            last_date = annotations[0].date
            path = self._cook_target_path(book_title, last_date)
            url = u"/notes/%s/" % path
            output += u"[%s](%s) (%d) _%s_\n\n" % (book_title, url, len(annotations), last_date)
        return output

    def _fetch_all(self, user_name):
        start = 0
        count = 100
        results = []
        while True:
            annotations = self._fetch_with_pagination(user_name, start, count)
            if not annotations:
                break
            start += count
            results.extend(annotations)
        return results

    def _collect_annotations_by_book(self, annotations):
        annotations_map = {}
        for annotation in annotations:
            annotations_map.setdefault(annotation.book_title, [])
            annotations_map[annotation.book_title].append(annotation)
        pairs = annotations_map.items()
        sorted_pairs = sorted(pairs, key=lambda (_, l): l[0].time, reverse=True)
        return sorted_pairs

    def _fetch_with_pagination(self, user_name, start, count=100):
        print '_fetch_with_pagination(%s, %s)' % (user_name, start)
        resp = requests.get('https://api.douban.com/v2/book/user/%s/annotations?count=%d&start=%d' % (user_name, count, start))
        resp.raise_for_status()
        data = resp.json()
        return [AnnotationDTO(d) for d in data['annotations']]

    def _cook_target_path(self, book_title, last_date):
        book_title = book_title.replace(' ', '-')
        book_title = book_title.replace('/', '-')
        path = u'%s-%s' % (last_date, book_title)
        return path


if __name__ == '__main__':
    exporter = AnnotationExporter()
    exporter.export('fleure')

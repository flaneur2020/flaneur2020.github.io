desc "re-generate this site."
task :default => :generate

task :g => :generate
task :generate do
  sh "bundle exec jekyll"
end

desc "start a local server to preview this site."
task :s => :server
task :server do
  sh "bundle exec jekyll --server --base-url --auto"
end

desc "create a new post file to _post"
task :n, :title do |task, args|
  title ||= args[:title] || ENV['t'] || ENV['T'] || ENV['TITLE'] || 'untitled'
  date_str = Time.now.strftime('%Y-%m-%d')
  path = File.dirname(__FILE__) + "/_posts/#{date_str}-#{title}.md"
  open(path, 'w+') do |f|
    f.write %{
      ---
      layout: post
      title: "#{title}"
      ---
    }.strip.split("\n").map{|s| s.strip }.join("\n")
  end
  puts "#{path}"
end

#  XML=~/Downloads/膜蛤のF叔的读书笔记.xml rake generate_notes && cat notes.md
desc "parse note.md from the exported xml from douban"
task :generate_notes, :xml do |task, args|
  require 'nokogiri'
  output = open(File.expand_path('../notes.md', __FILE__), 'w')
  output.puts '---'
  output.puts 'layout: paper'
  output.puts 'title: Notes'
  output.puts '---'
  output.puts
  path = args.fetch(:xml) || ENV.fetch('XML')
  doc = Nokogiri::XML(open(path))
  doc.css('book').each do |book|
    output.puts "## #{book['title']}"
    output.puts
    book.css('annotation').each do |annotation|
      title = annotation.css('title').first.content.split(/的笔记-/)[-1]
      content = annotation.css('content').first.content
      content = content.gsub(/<原文开始>(.*?)<\/原文结束>/m) do |m|
        m.gsub(/<原文开始>|<\/原文结束>/, '').lines.map{|l| "> #{l}" } * ""
      end
      output.puts "### #{title}"
      output.puts 
      output.puts content
      output.puts
    end
  end
end

desc "open the last blog post in gvim"
task :last do
  puts "#{Dir['_posts/*.md'].sort.last}"
end

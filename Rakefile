desc "re-generate this site."
task :default => :generate

task :g => :generate
task :generate do
  sh "jekyll"
end

desc "start a local server to preview this site."
task :s => :server
task :server do
  sh "jekyll --server --base-url --auto"
end

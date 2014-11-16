---
layout: post
title: "Note on Capistrano with Supervisor"
---

最近上线项目中尝试用 supervisord 监视 puma 进程。社区里教程似乎比较少，中间踩了个小坑在这里记录一下。

## Why Supervisord ?

supervisord 是 python 社区里比较常用的工具，能够保证进程在极端情况下挂掉之后重启，使 web app 成为一个默认活着的服务。诚然 puma / unicorn 这类 App 服务器的 master 进程已经足够稳定了，但是不能排除云服务商半夜重启服务器的可能性。

相比 daemontools，似乎比较容易配置；相比 systemd，兼容性更好，如果选用的镜像不是默认 systemd，这类核心的系统组件切换起来风险应该不小。

## Users

操作 supervisord 需要 sudo 权限，而 app 要降权执行，为此准备两个用户：

*deploy 用户*：

- 用以存放 Web App 代码，拥有 sudo 权限，且 NOPASSWD，用以控制 supervisorctl；
- 主用户组为 deploy，可以保证部署的代码文件的用户组为 deploy；
- 代码存放在 /home/deploy/www/myapp；

*app 用户*：

- 没有 sudo 权限，甚至不需要 home 目录，用以降权执行 App 代码；
- 加入 deploy 用户组，/home/deploy/www/myapp/shared 下的 tmp 目录和 log 目录需要 chmod -R g+w；

## supervisor.conf

假设项目中有三个 daemon：

```
[program:myapp-puma]
command=/home/deploy/.rbenv/shims/bundle exec puma -e production -S tmp/pids -C config/puma.rb
stdout_logfile=/home/deploy/www/myapp/shared/log/puma-1-out.log
stderr_logfile=/home/deploy/www/myapp/shared/log/puma-1-err.log
autostart=true
autorestart=true
stopsignal=QUIT
logfile_maxbytes=8mb
user=app
directory=/home/deploy/www/myapp/current
environment=RAILS_ENV="production",RBENV_VERSION="2.1.1",RBENV_ROOT="/home/deploy/.rbenv",PATH="/home/app/.rbenv/bin:/home/app/.rbenv/shims:/home/app/.rbenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/ usr/bin:/sbin:/bin"

[program:myapp-sidekiq]
command=/home/deploy/.rbenv/shims/bundle exec sidekiq -e production
stdout_logfile=/home/deploy/www/myapp/shared/log/sidekiq-1-out.log
stderr_logfile=/home/deploy/www/myapp/shared/log/sidekiq-1-err.log
autostart=true
autorestart=true
stopsignal=QUIT
logfile_maxbytes=8mb
user=app
directory=/home/deploy/www/myapp/current
environment=RAILS_ENV="production",RBENV_VERSION="2.1.1",RBENV_ROOT="/home/deploy/.rbenv",PATH="/home/app/.rbenv/bin:/home/app/.rbenv/shims:/home/app/.rbenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/ usr/bin:/sbin:/bin"

[program:myapp-clockwork]
command=/home/deploy/.rbenv/shims/bundle exec clockwork config/clockwork.rb
stdout_logfile=/home/deploy/www/myapp/shared/log/clockwork-1-out.log
stderr_logfile=/home/deploy/www/myapp/shared/log/clockwork-1-err.log
autostart=true
autorestart=true
stopsignal=QUIT
logfile_maxbytes=8mb
user=app
directory=/home/deploy/www/myapp/current
environment=RAILS_ENV="production",RBENV_VERSION="2.1.1",RBENV_ROOT="/home/deploy/.rbenv",PATH="/home/app/.rbenv/bin:/home/app/.rbenv/shims:/home/app/.rbenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/ usr/bin:/sbin:/bin"
[group:myapp]
programs=myapp-puma,myapp-sidekiq,myapp-clockwork
```

有个 trick 的地方是：command 配置中的可执行文件的路径似乎并不会参考环境变量 PATH 的影响，所以最好有完整的执行路径。不过执行时会受到 environment 中配置的环境变量影响，配置 RBENV_VERSION="2.1.1",RBENV_ROOT="/home/deploy/.rbenv" 就可以选择正确的 ruby 版本了。

这个配置可以放在代码仓库的 config/supervisord.conf 下，每次部署时将它拷贝到 /etc/supervisor/conf.d/myapp.conf，然后 supervisorctl reread / update 重新装载配置。

## deploy.rb

```
lock '3.2.1'

set :application, 'myapp'
set :repo_url, 'git@github.com:myapp/myapp.git'
set :branch, fetch(:branch, "master")
set :deploy_to, '/home/deploy/www/myapp'
set :deploy_user, 'deploy'

set :rbenv_type, :user
set :rbenv_ruby, '2.1.1'
set :rbenv_map_bins, %w{rake gem bundle ruby rails}

set :scm, :git
set :format, :pretty
set :log_level, :debug
set :pty, true
set :keep_releases, 5

set :linked_files, %w{config/settings.yml}
set :linked_dirs, %w{bin log tmp/pids tmp/cache tmp/sockets vendor/bundle public/system}

# Default value for default_env is {}
# set :default_env, { path: "/opt/ruby/bin:$PATH" }

namespace :supervisord do
  task :export do
    on roles(:app), in: :sequence do
      sudo "cp #{release_path}/config/supervisord.conf /etc/supervisor/conf.d/myapp.conf"
      sudo "supervisorctl reread"
      sudo "supervisorctl update"
      sudo "supervisorctl status"
    end
  end
end


namespace :deploy do
  desc 'Restart application'
  task :restart do
    on roles(:app), in: :sequence, wait: 5 do
      sudo "supervisorctl restart myapp:myapp-clockwork"
      sudo "supervisorctl restart myapp:myapp-sidekiq"
      # 通过发送 SIGUSR1 触发 puma 的 phased-restart
      sudo "supervisorctl start myapp:myapp-puma"
      sudo "kill -SIGUSR1 $(cat #{shared_path}/tmp/pids/puma.pid)"
    end
  end

  task :chmod do
    # 确保 tmp 目录和 log 目录有对同用户组的写权限
    on roles(:app), in: :sequence, wait: 5 do
      sudo "chmod -R g+w #{shared_path}/log"
      sudo "chmod -R g+w #{shared_path}/tmp"
    end
  end

  after :publishing,
    'supervisord:export',
    'deploy:chmod'
    'deploy:restart'
end
```

capistrano 配合几个常见的插件（如 capistrano-rails, capistrano-rbenv, capistrano-rbenv-install, capistrano-bundler 等 ），即已满足基本的部署需求：安装 ruby；建立部署的目录结构；链接共享文件或目录；部署代码；安装依赖；跑 Asset Pipeline；跑 Migration。剩下唯一需要自己做的就是提供重启、确保 supervisor 配置、确保文件写权限了。

puma 允许通过 SIGUSR1 信号触发 phased-restart， 做到在重启时不中断服务。但 supervisor 并没有提供发送信号的工具命令，仍需要手工发送信号。所以在配置里需要做一个 work around：首先通过  sudo "supervisorctl start myapp:myapp-puma" 确保 puma 活着，然后读取 tmp/pids/puma.pid 发送 SIGUSR1。

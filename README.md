(base) caoying@lthpc:/data/AIPD$ conda create -n ProtSATT python=3.10.12
2 channel Terms of Service accepted
Retrieving notices: done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): failed

# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/exception_handler.py", line 30, in __call__
        return func(*args, **kwargs)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/cli/main.py", line 53, in main_subshell
        exit_code = do_call(args, parser)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/cli/conda_argparse.py", line 190, in do_call
        result = getattr(module, func_name)(args, parser)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/notices/core.py", line 137, in wrapper
        return_value = func(*args, **kwargs)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/cli/main_create.py", line 215, in execute
        install(args, parser, "create")
        ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/cli/install.py", line 344, in install
        unlink_link_transaction = solver.solve_for_transaction(
            deps_modifier=deps_modifier,
        ...<4 lines>...
            ),
        )
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/core/solve.py", line 164, in solve_for_transaction
        unlink_precs, link_precs = self.solve_for_diff(
                                   ~~~~~~~~~~~~~~~~~~~^
            update_modifier,
            ^^^^^^^^^^^^^^^^
        ...<5 lines>...
            should_retry_solve,
            ^^^^^^^^^^^^^^^^^^^
        )
        ^
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/core/solve.py", line 233, in solve_for_diff
        final_precs = self.solve_final_state(
            update_modifier,
        ...<4 lines>...
            should_retry_solve,
        )
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda_libmamba_solver/solver.py", line 171, in solve_final_state
        index = self._collect_all_metadata(
            channels=channels,
        ...<2 lines>...
            in_state=in_state,
        )
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/common/io.py", line 109, in decorated
        return f(*args, **kwds)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda_libmamba_solver/solver.py", line 270, in _collect_all_metadata
        index = LibMambaIndexHelper(
            channels=[*conda_build_channels, *channels],
        ...<8 lines>...
            build_repodata_subset=self._build_repodata_subset,
        )
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda_libmamba_solver/index.py", line 279, in __init__
        self.repos: list[_ChannelRepoInfo] = self._load_channels()
                                             ~~~~~~~~~~~~~~~~~~~^^
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda_libmamba_solver/index.py", line 385, in _load_channels
        channel_repos_info = self._load_channel_repo_info_shards(urls_to_channel)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda_libmamba_solver/index.py", line 403, in _load_channel_repo_info_shards
        channel_data = self.build_repodata_subset(root_packages, urls_to_channel)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/_private/shards/subset.py", line 578, in build_repodata_subset
        channel_data = fetch_channels(channels)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/_private/shards/shards.py", line 772, in fetch_channels
        found = future.result()
      File "/data/AIPD/miniconda3/lib/python3.14/concurrent/futures/_base.py", line 447, in result
        return self.__get_result()
               ~~~~~~~~~~~~~~~~~^^
      File "/data/AIPD/miniconda3/lib/python3.14/concurrent/futures/_base.py", line 396, in __get_result
        raise self._exception
      File "/data/AIPD/miniconda3/lib/python3.14/concurrent/futures/thread.py", line 86, in run
        result = ctx.run(self.task)
      File "/data/AIPD/miniconda3/lib/python3.14/concurrent/futures/thread.py", line 73, in run
        return fn(*args, **kwargs)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/_private/shards/shards.py", line 625, in fetch_shards_index
        with repo_cache.lock("r+") as state_file:
             ~~~~~~~~~~~~~~~^^^^^^
      File "/data/AIPD/miniconda3/lib/python3.14/contextlib.py", line 141, in __enter__
        return next(self.gen)
      File "/data/AIPD/miniconda3/lib/python3.14/site-packages/conda/gateways/repodata/__init__.py", line 686, in lock
        with self.cache_path_state.open(mode) as state_file, lock(state_file):
             ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
      File "/data/AIPD/miniconda3/lib/python3.14/pathlib/__init__.py", line 771, in open
        return io.open(self, mode, buffering, encoding, errors, newline)
               ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    PermissionError: [Errno 13] Permission denied: '/data/AIPD/miniconda3/pkgs/cache/47929eba.info.json'

`$ /data/AIPD/miniconda3/bin/conda create -n ProtSATT python=3.10.12`

  environment variables:
                 CIO_TEST=<not set>
        CONDA_DEFAULT_ENV=base
                CONDA_EXE=/data/AIPD/miniconda3/bin/conda
             CONDA_PREFIX=/data/AIPD/miniconda3
    CONDA_PROMPT_MODIFIER=(base)
         CONDA_PYTHON_EXE=/data/AIPD/miniconda3/bin/python
               CONDA_ROOT=/data/AIPD/miniconda3
              CONDA_SHLVL=1
           CURL_CA_BUNDLE=<not set>
               LD_PRELOAD=<not set>
                     PATH=/data/caoying/.vscode-server/data/User/globalStorage/github.copilot-
                          chat/debugCommand:/data/caoying/.vscode-
                          server/data/User/globalStorage/github.copilot-
                          chat/copilotCli:/data/caoying/.vscode-server/cli/servers/Stable-
                          4fe60c8b1cdac1c4c174f2fb180d0d758272d713/server/bin/remote-cli:/data/A
                          IPD/miniconda3/bin:/data/AIPD/miniconda3/condabin:/usr/local/sbin:/usr
                          /local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/
                          snap/bin
       REQUESTS_CA_BUNDLE=<not set>
            SSL_CERT_FILE=<not set>

     active environment : base
    active env location : /data/AIPD/miniconda3
            shell level : 1
       user config file : /data/caoying/.condarc
 populated config files : /data/AIPD/miniconda3/.condarc
                          /data/AIPD/miniconda3/condarc.d/anaconda-auth.yml
          conda version : 26.5.3
    conda-build version : not installed
         python version : 3.14.6.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=zen4
                          __conda=26.5.3=0
                          __cuda=12.8=0
                          __glibc=2.35=0
                          __linux=6.8.0=0
                          __unix=0=0
       base environment : /data/AIPD/miniconda3  (writable)
      conda av data dir : /data/AIPD/miniconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /data/AIPD/miniconda3/pkgs
                          /data/caoying/.conda/pkgs
       envs directories : /data/AIPD/miniconda3/envs
                          /data/caoying/.conda/envs
    temporary directory : /tmp
               platform : linux-64
             user-agent : conda/26.5.3 requests/2.34.2 CPython/3.14.6 Linux/6.8.0-52-generic ubuntu/22.04.1 glibc/2.35 solver/libmamba conda-libmamba-solver/26.7.0 libmambapy/2.3.2 aau/0.8.1 c/. s/. e/.
                UID:GID : 1010:1011
             netrc file : None
           offline mode : False


An unexpected error has occurred. Conda has prepared the above report.
If you suspect this error is being caused by a malfunctioning plugin,
consider using the --no-plugins option to turn off plugins.

Example: conda --no-plugins install <package>

Alternatively, you can set the CONDA_NO_PLUGINS environment variable on
the command line to run the command without plugins enabled.

Example: CONDA_NO_PLUGINS=true conda install <package>

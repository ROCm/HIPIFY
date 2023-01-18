import os
from typing import Union
from git import Repo, Remote, RemoteReference
from pathlib import Path

def format_toc(repo_path: Union[str, os.PathLike, None] = None):
    pwd = Path(__file__).resolve().parent
    if repo_path is None:
        repo_path = pwd.parent
    at_start = True
    repo = Repo(repo_path, search_parent_directories=True)
    assert not repo.bare
    try:
        branch = repo.active_branch.name
    except TypeError as exc:  # HEAD is detached commit
        checked_heads = []
        for head in repo.heads:
            checked_heads.append(head.name)
            if head.commit == repo.head.commit:
                branch = head.name
                break
        else:  # loop fell through
            for remote in repo.remotes:
                remote: Remote
                for ref in remote.refs:
                    ref: RemoteReference
                    if ref.commit == repo.head.commit:
                        branch = ref.name.split('/')[-1]
                        break
                else:  # loop fell through
                    raise TypeError("A branch name could not be determined.\n(Checked heads: %s)" % ' '.join(checked_heads)) from exc
    with open(pwd / '_toc.yml.in', 'r', encoding='utf-8') as input:
        with open(pwd / '_toc.yml', 'w', encoding='utf-8') as output:
            for line in input.readlines():
                if line[0] == '#' and at_start:
                    continue
                at_start = False
                output.write(line.format(branch=branch))


if __name__ == '__main__':
    format_toc()

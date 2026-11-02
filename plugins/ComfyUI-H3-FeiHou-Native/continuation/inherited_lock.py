"""Reuse the supervisor's open-file description in native job_runner.main."""
import os
from pathlib import Path


class InheritedRuntimeLock:
    def __init__(self,path,fd):
        self.path=Path(path)
        self.fd=int(fd)
        actual,expected=os.fstat(self.fd),self.path.stat()
        if self.fd<0 or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise RuntimeError('Invalid inherited runtime lock')

    def __fspath__(self):return os.fspath(self.path)
    def __str__(self):return str(self.path)
    def __getattr__(self,name):return getattr(self.path,name)
    def resolve(self,*args,**kwargs):return self.path.resolve(*args,**kwargs)
    def stat(self,*args,**kwargs):return self.path.stat(*args,**kwargs)
    def open(self,mode='w',*args,**kwargs):
        # dup shares the locked open-file description, so native LOCK_NB is
        # not a competing acquisition. Closing the worker copy leaves the
        # supervisor's copy locked until all descendants have exited.
        return os.fdopen(os.dup(self.fd),mode)

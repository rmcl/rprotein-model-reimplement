from invoke import task

@task
def test(c):
    c.run("python -m pytest", pty=True)

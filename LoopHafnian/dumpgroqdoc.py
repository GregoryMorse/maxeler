def dump_groq_doc():
    import os, pydoc, pkgutil, groq, groq_api, importlib, functools, shutil
    try: os.mkdir("groqdoc")
    except FileExistsError: pass
    os.chdir("groqdoc")
    def import_submodules(package, recursive=True):
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        paths = [x for x in package.__path__ if x.startswith('/opt/groq/runtime/site-packages/groq')]
        for loader, name, is_pkg in pkgutil.walk_packages(paths):
            full_name = package.__name__ + '.' + name
            try:
                results[full_name] = importlib.import_module(full_name)
            except ModuleNotFoundError: continue
            except FileNotFoundError: continue
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
        return results    
    for module in import_submodules(groq):
        if not module.startswith("groq."): continue
        attr = functools.reduce(lambda x, y: getattr(x, y), [groq] + module.split('.')[1:])
        setattr(attr, "__all__", dir(attr))
        pydoc.writedoc(module)
    groq.__all__ = dir(groq)
    pydoc.writedoc(groq)
    groq_api.__all__ = dir(groq_api)
    pydoc.writedoc(groq_api)
    os.chdir("..")
    shutil.make_archive("groqdoc", 'zip', "groqdoc")
dump_groq_doc()

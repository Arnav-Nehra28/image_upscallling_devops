import pkgutil, torchvision
print('torchvision', torchvision.__file__)
import torchvision.transforms as T
print('transforms package:', T.__file__)
print('submodules:')
for m in pkgutil.iter_modules(T.__path__):
    print(' -', m.name)

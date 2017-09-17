# begin
# end
import inspect

def parent_fun(arg1, arg2, arg3):
    data =child_fun()
    print(data)

    return 0

def child_fun():
    data = FUN('parent_fun')
    pass
    import pdb; pdb.set_trace()
    return 'helo world'

parent_fun(1,2,3)


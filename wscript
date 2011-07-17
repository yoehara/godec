VERSION = '0.0.1'
APPNAME = 'godec'

top = '.'
out = 'build'

def options(ctx):
  ctx.tool_options('compiler_cxx')
                                                                                                          
def configure(ctx):
  ctx.check_tool('compiler_cxx')
  ctx.env.append_value('std', ['c++0x'])
  ctx.env.CXXFLAGS += ['-O2', '-std=c++0x', '-Wno-c++0x-compat',  '-DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET']
#'-Wall', '-W', '-Wno-c++0x-compat', 

def build(bld):
  bld.recurse('src/')

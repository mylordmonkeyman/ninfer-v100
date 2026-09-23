add_library(ninfer_xgrammar STATIC
  cpp/compiled_grammar.cc cpp/config.cc cpp/earley_parser.cc cpp/fsm_builder.cc cpp/fsm.cc
  cpp/grammar_builder.cc cpp/grammar_compiler.cc cpp/grammar_functor.cc cpp/grammar_matcher.cc
  cpp/grammar_parser.cc cpp/grammar_printer.cc cpp/grammar.cc cpp/json_schema_converter_ext.cc
  cpp/json_schema_converter.cc cpp/lark_converter.cc cpp/regex_converter.cc cpp/structural_tag.cc
  cpp/testing.cc cpp/tokenizer_info.cc cpp/support/logging.cc cpp/support/recursion_guard.cc)
target_include_directories(ninfer_xgrammar SYSTEM PUBLIC include 3rdparty/dlpack/include
  PRIVATE 3rdparty/picojson)
target_compile_definitions(ninfer_xgrammar PRIVATE XGRAMMAR_ENABLE_CPPTRACE=0 XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
target_link_libraries(ninfer_xgrammar PUBLIC Threads::Threads)

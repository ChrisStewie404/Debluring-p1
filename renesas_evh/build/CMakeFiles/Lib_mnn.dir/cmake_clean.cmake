file(REMOVE_RECURSE
  "libLib_mnn.a"
  "libLib_mnn.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/Lib_mnn.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

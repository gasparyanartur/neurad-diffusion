src_dir=${SRC_DIR:-"x_artga@berzelius.nsc.liu.se:/home/x_artga/dev/neurad-studio/outputs/"}
dst_dir=${DST_DIR:-"/home/gasparyanartur/dev/outputs/"}
rel_dir=${REL_DIR}
if [ -z "$rel_dir" ]; then
    echo "Relative directory is not set"
    exit 1
fi

full_src_dir=$src_dir$rel_dir
full_dst_dir=$dst_dir$rel_dir
# make full_dst_dir_parent
dst_dir_parent=$(dirname $full_dst_dir)
mkdir -p $dst_dir_parent

rsync -avrzh --progress $full_src_dir $full_dst_dir
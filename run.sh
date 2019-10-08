for f in test_programs/*.py; do
  python altered_dis.py "$f" "-f" > "$f.dis";
done

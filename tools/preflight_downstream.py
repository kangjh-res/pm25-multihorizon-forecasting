"""Read-only check of known downstream prerequisites. No notebook execution."""
import argparse,csv,pathlib,sys

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--research-root',type=pathlib.Path,required=True)
 p.add_argument('--manifest',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'V47_IO_DEPENDENCIES.csv')
 args=p.parse_args(); root=args.research_root.expanduser().resolve()
 if not root.is_dir(): p.error('Research root does not exist: '+str(root))
 with args.manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 print('READ-ONLY STATIC PREFLIGHT; file names may be inputs OR outputs')
 for row in rows:
  names=[n for n in row['referenced_file_names'].split(';') if n]
  found=[];missing=[]
  for n in names:
   matches=list(root.rglob(n))
   (found if matches else missing).append(n)
  print(f"{row['script']}: existing={len(found)}, absent={len(missing)}; absent names: {', '.join(missing[:12])}")
 print('No code was executed, no files were written, and absence is NOT a failure because names include outputs.')
 return 0
if __name__=='__main__':sys.exit(main())

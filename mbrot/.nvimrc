augroup rsyn_after_save
    au BufWritePost * !rsync -avz % nathanpro@linux.ime.usp.br:~/03/
augroup END

echom "rsync loaded"
echom "nathanpro@linux.ime.usp.br:~/03/"

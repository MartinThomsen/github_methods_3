for(i in 1:length(df$trial.type)) {
  if (df$obj.resp[i]== 'o' & df$target.type[i] == 'odd')
  {
    df$correct[i] == 1
  }
  else{
    df$correct[i] == 0
  }
}

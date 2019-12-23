## The original code is bellow
## https://qiita.com/dosec/items/c6aef40fae6977fd89ab

TRANSUNIT <- hash(c("十", "百", "千"), c(10,100,1000))
TRANSMANS <-　hash( "万", 10000)
re_suji <- '[十拾百千万億兆\\d]+'
re_kunit = '[十拾百千]|\\d+'
re_manshin = '[万億兆]|[^万億兆]+'

transvalue <- function(sj, re_obj, transdic){
    unit <- 1
    result <- 0
    for (piece in rev(str_extract_all(sj, pattern=re_obj)[[1]])){
        if (any(str_detect(keys(transdic), piece))){
            if (unit > 1){
                result <- result + unit 
            }
            unit <- transdic[[piece]]
        }else{
            if (is.integer(type.convert(piece))){
                val <- type.convert(piece)
            }else{
                val <- transvalue(piece, re_kunit, TRANSUNIT)
            }
            result <- result + (val * unit)
            unit <- 1
        }            
    }
    if (unit > 1){
        result <- result + unit
    }
    return(result)
}

kanji2arabic <- function(text) {
    tt_ksuji <- c('一'='1', '二'='2', '三'='3', '四'='4', '五'='5', 
                  '六'='6', '七'='7', '八'='8', '九'='9', '〇'='0')

    transuji <- str_replace_all(text, tt_ksuji)  # 漢数字の変換 
    for (suji in rev(str_extract_all(transuji, pattern=re_suji)[[1]])){
        if (!(is.integer(type.convert(suji)))){
            arabic <- transvalue(suji, re_manshin, TRANSMANS)
            transuji <- str_replace_all(transuji, suji, as.character(arabic))
        }
    }
    return(transuji)
}

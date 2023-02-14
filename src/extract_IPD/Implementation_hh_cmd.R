suppressPackageStartupMessages(library(tidyverse))

# change folder names if files are in different location
setwd("/Users/haeunhwangbo/Dropbox/PalmerLab/clinical-additivity/src/extract_IPD/")

source("IPD_Functions.R")
library(stringr)
# get input from command line
args = commandArgs(trailingOnly = T)
# directory that contains digitized curve and at-risk table (doesn't end with /)
dir = args[1]
# file name excluding extension (.csv)
mono_prefix = args[2]  # ex) Lung_Carboplatin+Etoposide_Horn2018_OS
comb_prefix = args[3]  # ex) Lung_Atezolizumab-Carboplatin+Etoposide_Horn2018_OS
comb_prefix = str_replace_all(comb_prefix, "[\r\n]" , "")

tokens = str_split(comb_prefix, "_", simplify = T)
testDrug = str_split(tokens[2], "-", simplify = T)[1]  # ex) Atezolizumab
# ex) Lung_Atezolumab_Horn2018_OS_at-risk
atrisk_file = paste(tokens[1], testDrug, tokens[3], tokens[4], "at-risk", sep='_')
print(atrisk_file)

tryCatch(
  expr = {
    DF <- read.csv(paste0(dir, '/', atrisk_file, '.csv'))
    
    for(arm in c('mono', 'comb')){
      
      if(arm == "mono"){
        AR <- data.frame(interval = seq(1, dim(DF)[1]), 
                         trisk = DF$time, nrisk = DF$control, TE = DF$control[1],
                         Arm = "control")
        fileprefix = mono_prefix
      }
      else{
        AR <- data.frame(interval = seq(1, dim(DF)[1]), 
                         trisk = DF$time, nrisk = DF$treatment, TE = DF$treatment[1],
                         Arm = "treatment")
        fileprefix = comb_prefix
      }
      
      TSD <- read.csv(paste0(dir, '/', fileprefix, ".csv"))
      digizeit <- DIGI.CLEANUP(TSD)
      
      
      # if survival is in percentage, convert it to 0-1
      if (max(digizeit$S) > 2){
        digizeit$S = digizeit$S / 100
      }
      
      pub.risk <- K.COORDINATES(AR, digizeit)
      
      IPD <- GENERATEINDIVIDUALDATA(tot.events = unique(AR$TE), 
                                    arm.id = unique(AR$Arm), 
                                    digizeit = digizeit, 
                                    pub.risk = pub.risk)
      IPD <- data.frame(Time=IPD$Time, Event=IPD$Event, 
                        Arm=IPD$Arm)
      nm <- paste(unique(AR$Slide),  paste(unique(AR$Arm), unique(AR$Subpop), sep='_'),sep='_')
      write.csv(IPD, paste0(dir, '/', fileprefix, "_indiv.csv"), row.names = FALSE, quote = F)
    }
  },
  
  warning = function(w){ 
    message(paste0("No at-risk file: ", dir, '/', atrisk_file, '.csv'))
  }
)



#IPD$Time <- as.numeric(as.character(IPD$Time))
#IPD$Event <- as.numeric(as.character(IPD$Event))
#survfit(Surv(Time, Event) ~ 1, data=IPD)
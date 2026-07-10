# Section 01: Prepare the R Environment. ----
# ************************************************************************************************************************
# Section 01: Prepare the R Environment.
# ************************************************************************************************************************
graphics.off();rm(list=ls());cat("\14")
pacman::p_load(Amelia,Boruta,caret,collapse,data.table,dplyr,doParallel,e1071,forecast,glmnet,h2o,haven,here,imputeTS,lars,leaps,lfe,lubridate,
               Metrics,mice,missForest,neuralnet,parallel,plm,plotly,plyr,psych,reshape,readxl,rprojroot,stargazer,stringr,tsbox,vars,zoo)

Directory <- getwd()

# Section 02: Common Formatting. ----
# ************************************************************************************************************************
# Section 02: Common Formatting.
# ************************************************************************************************************************
MyColors <- c("#FFFFCC","#FFEDA0","#FED976","#FEB24C","#FD8D3C","#FC4E2A","#E31A1C","#B10026","#F7FBFF","#DEEBF7",
              "#C6DBEF","#9ECAE1","#6BAED6","#4292C6","#2171B5","#084594","#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B")

MyColors3 <- c("#F7FBFF","#A6D854","#2166AC")

MyColors2 <- c("#A6D854","#2166AC")

Theme <-   theme(legend.position="bottom",
                 legend.direction="horizontal",
                 legend.spacing.x=unit(0.5,'cm'),
                 legend.spacing.y=unit(0.2,'cm'),
                 legend.text=element_text(size=10,family="serif"),
                 legend.title=element_blank(),
                 axis.text.x=element_text(family="serif",color="black",size=11),
                 axis.title.x=element_blank(),
                 axis.title.y=element_text(angle=90,vjust=0.5,family="serif",size=11),
                 axis.text.y=element_text(angle=0,vjust=0.5,family="serif"),
                 axis.line=element_line(colour="black",size=1,linetype="solid"),
                 panel.grid=element_line(size=1,color="black"),
                 panel.grid.major.x=element_line(size=.05,color="grey",linetype=2),
                 panel.grid.major.y=element_line(size=.05,color="grey",linetype=2),
                 panel.background=element_blank(),
                 panel.border=element_rect(fill=NA,colour="black",size=1,linetype="solid"),
                 plot.title=element_text(size=11,color="black",family="serif",hjust=0.5),
                 plot.subtitle=element_text(size=11,color="black",family="serif"),
                 plot.caption=element_text(size=11,color="black",hjust=0,family="serif"),
                 strip.text.x=element_text(size=11,family="serif"))

# Annexure 04B: SHY Shocks. ----
# ************************************************************************************************************************
# Annexure 04B: SHY Shocks.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A04B_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Sector",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Sector)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Reduction in Productivity") +
  labs(title=paste0("Annexure 04B: Percentage Change in Productivity in the First Year under the First Scenario"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_04B.png"),width=29.7,height=21,units="cm")

# Annexure 05A: Consumption Shocks. ----
# ************************************************************************************************************************
# Annexure 05A: Consumption Shocks.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A05A_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Scenario",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Scenario)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Reduction in Consumption") +
  labs(title=paste0("Annexure 05A: Change in Consumption (%GDP) in the First Year under the Scenarios"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_05A.png"),width=29.7,height=21,units="cm")

# Annexure 05B: Index of Risk Aversion. ----
# ************************************************************************************************************************
# Annexure 05B: Index of Risk Aversion.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A05B_Data",skip=1)
names(Data)[2] <- "Value"

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value)) +
  geom_bar(stat="identity",position="dodge",fill="#4292C6") +
  Theme +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  labs(title=paste0("Annexure 05B: Index of Risk Aversion relative to the US"),caption="Source: Gandelman and Hernández-Murillo (2014).")
ggsave(paste0(Directory,"/Annexure_05B.png"),width=29.7,height=21,units="cm")

# Annexure 05C: Risk Premium on Human Wealth. ----
# ************************************************************************************************************************
# Annexure 05C: Risk Premium on Human Wealth.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A05C_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Scenario",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Scenario)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Point Increase in Risk Premium") +
  labs(title=paste0("Annexure 05C: Change in Risk Premium on Human Wealth for the First Year under the Scenarios"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_05C.png"),width=29.7,height=21,units="cm")

# Annexure 06A: Index of Country Risk. ----
# ************************************************************************************************************************
# Annexure 06A: Index of Country Risk.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A06A_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Component",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Component)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("Value") +
  labs(title=paste0("Annexure 06A: Index of Country Risk and its Components"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_06A.png"),width=29.7,height=21,units="cm")

# Annexure 06B: Country Risk Premium. ----
# ************************************************************************************************************************
# Annexure 06B: Country Risk Premium.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A06B_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Scenario",value.name="Value")
Data <- Data[which(Data$Region != "USA"),]
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Scenario)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Point Increase in Country Risk Premium") +
  labs(title=paste0("Annexure 06B: Change in Country Risk Premium in the First Year under the Scenarios"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_06B.png"),width=29.7,height=21,units="cm")

# Annexure 06C: Sector Risk Premium. ----
# ************************************************************************************************************************
# Annexure 06C: Sector Risk Premium.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A06C_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Sector",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Sector)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Point Increase in Sector Risk Premium") +
  labs(title=paste0("Annexure 06C: Change in Sector Risk Premium in the First Year under the Scenarios"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_06C.png"),width=29.7,height=21,units="cm")

# Annexure 07A: Government Expenditure. ----
# ************************************************************************************************************************
# Annexure 07A: Government Expenditure.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A07A_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Component",value.name="Value")
Data$Value <- as.numeric(as.character(Data$Value))
Data <- na.omit(Data)
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Component)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% GDP") +
  labs(title=paste0("Annexure 07A: Change in Government Expenditure"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_07A.png"),width=29.7,height=21,units="cm")

# Annexure 07B: Government Expenditure Allocation. ----
# ************************************************************************************************************************
# Annexure 07B: Government Expenditure Allocation.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A07B_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Sector",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Sector)) +
  geom_bar(stat="identity",position="stack") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Allocation") +
  labs(title=paste0("Annexure 07B: General Government Expenditure Allocation across Sectors"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_07B.png"),width=29.7,height=21,units="cm")

# Annexure 07C: Government Expenditure across Sectors. ----
# ************************************************************************************************************************
# Annexure 07C: Government Expenditure across Sectors.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A07C_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Sector",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Sector)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% GDP") +
  labs(title=paste0("Annexure 07C: General Government Expenditure across Sectors"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_07C.png"),width=29.7,height=21,units="cm")

# Annexure 07D: Wage Subsidy. ----
# ************************************************************************************************************************
# Annexure 07D: Wage Subsidy.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="A07D_Data",skip=1)
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Sector",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")

Plot <- 
  ggplot(data=Data,mapping=aes(x=Region,y=Value,fill=Sector)) +
  geom_bar(stat="identity",position="dodge") +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("Value") +
  labs(title=paste0("Annexure 07D: Wage Subsidy"),caption="Source: Constructed by the Authors.")
ggsave(paste0(Directory,"/Annexure_07D.png"),width=29.7,height=21,units="cm")

# Figure 01 & 02: Dynamic Macro Results. ----
# ************************************************************************************************************************
# Figure 01 & 02: Dynamic Macro Results.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="Figure_1-4_Data",skip=1)
Data <- Data[which(!Data$Label %in% c("Change in Output","Change in Sectoral Employment")),]
Data <- subset(Data,select=-c(Variable,Unit))
Data <- reshape2::melt(Data,id.vars=c("Scenario","Label"),variable.name="Year",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")
Labels <- unique(Data$Label)

Plot <- 
  ggplot(data=Data[which(Data$Label %in% Labels[1:4]),],mapping=aes(x=Year,y=Value,fill=Scenario)) +
  geom_line(aes(linetype=Scenario,group=Scenario,color=Scenario),linewidth=1.5) +
  geom_point(aes(shape=Scenario,color=Scenario),size=2) +
  scale_color_manual(values=MyColors) + 
  scale_linetype_manual(values=c(1,2,3,4,5,6)) +
  scale_shape_manual(values=c(4,8,15,16,17,18)) +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_y_continuous(expand=c(0,0)) +
  scale_x_discrete(expand=c(0.01,0.01)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Deviation from Baseline") +
  facet_wrap(~Label,nrow=2,ncol=2) +
  theme(strip.text.x=element_text(size=11,family="serif"),panel.spacing=unit(2,"lines")) +
  labs(title=paste0("Figure 01: Dynamic Macroeconomic Results for Australia"),caption="Source: Constructed by the Author using G-Cubed (GGG6G) Simulations (2020).")
ggsave(paste0(Directory,"/Figure01.png"),width=29.7,height=21,units="cm")

Plot <- 
  ggplot(data=Data[which(Data$Label %in% Labels[5:8]),],mapping=aes(x=Year,y=Value,fill=Scenario)) +
  geom_line(aes(linetype=Scenario,group=Scenario,color=Scenario),linewidth=1.5) +
  geom_point(aes(shape=Scenario,color=Scenario),size=2) +
  scale_color_manual(values=MyColors) + 
  scale_linetype_manual(values=c(1,2,3,4,5,6)) +
  scale_shape_manual(values=c(4,8,15,16,17,18)) +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_y_continuous(expand=c(0,0)) +
  scale_x_discrete(expand=c(0.01,0.01)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Deviation from Baseline") +
  facet_wrap(~Label,nrow=2,ncol=2) +
  theme(strip.text.x=element_text(size=11,family="serif"),panel.spacing=unit(2,"lines")) +
  labs(title=paste0("Figure 02: Dynamic Macroeconomic Results for Australia (Contd.)"),caption="Source: Constructed by the Author using G-Cubed (GGG6G) Simulations (2020).")
ggsave(paste0(Directory,"/Figure02.png"),width=29.7,height=21,units="cm")

# Figure 03 & 04: Dynamic Sector Results. ----
# ************************************************************************************************************************
# Figure 03 & 04: Dynamic Sector Results.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="Figure_1-4_Data",skip=1)
Data <- Data[which(Data$Label %in% c("Change in Output","Change in Sectoral Employment")),]
Data$Sector <- substr(Data$Variable,4,5)
Data$Sector <- plyr::mapvalues(Data$Sector,from=unique(Data$Sector),to=c("Energy","Mining","Agriculture","Durable Manufacturing","Non-durable Manufacturing","Services"))
Data <- subset(Data,select=-c(Variable,Unit))
Data <- reshape2::melt(Data,id.vars=c("Scenario","Label","Sector"),variable.name="Year",value.name="Value")
MyColors <- c("#FD8D3C","#FED976","#A1D99B","#9ECAE1","#4292C6","#084594")
Labels <- unique(Data$Label)

Plot <- 
  ggplot(data=Data[which(Data$Label %in% Labels[1]),],mapping=aes(x=Year,y=Value,fill=Scenario)) +
  geom_line(aes(linetype=Scenario,group=Scenario,color=Scenario),linewidth=1.5) +
  geom_point(aes(shape=Scenario,color=Scenario),size=2) +
  scale_color_manual(values=MyColors) + 
  scale_linetype_manual(values=c(1,2,3,4,5,6)) +
  scale_shape_manual(values=c(4,8,15,16,17,18),guide=guide_legend(override.aes=list(fill=MyColors))) +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_y_continuous(expand=c(0,0)) +
  scale_x_discrete(expand=c(0.01,0.01)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Deviation from Baseline") +
  facet_wrap(~Sector,nrow=2,ncol=3) +
  theme(strip.text.x=element_text(size=11,family="serif"),panel.spacing=unit(2,"lines")) +
  labs(title=paste0("Figure 03: Dynamic Sectoral Results for Australia"),caption="Source: Constructed by the Author using G-Cubed (GGG6G) Simulations (2020).")
ggsave(paste0(Directory,"/Figure03.png"),width=29.7,height=21,units="cm")

Plot <- 
  ggplot(data=Data[which(Data$Label %in% Labels[2]),],mapping=aes(x=Year,y=Value,fill=Scenario)) +
  geom_line(aes(linetype=Scenario,group=Scenario,color=Scenario),linewidth=1.5) +
  geom_point(aes(shape=Scenario,color=Scenario),size=2) +
  scale_color_manual(values=MyColors) + 
  scale_linetype_manual(values=c(1,2,3,4,5,6)) +
  scale_shape_manual(values=c(4,8,15,16,17,18),guide=guide_legend(override.aes=list(fill=MyColors))) +
  Theme +
  scale_fill_manual(values=MyColors) +
  scale_y_continuous(expand=c(0,0)) +
  scale_x_discrete(expand=c(0.01,0.01)) +
  guides(fill=guide_legend(nrow=1)) +
  ylab("% Deviation from Baseline") +
  facet_wrap(~Sector,nrow=2,ncol=3) +
  theme(strip.text.x=element_text(size=11,family="serif"),panel.spacing=unit(2,"lines")) +
  labs(title=paste0("Figure 04: Dynamic Sectoral Results for Australia (Contd.)"),caption="Source: Constructed by the Author using G-Cubed (GGG6G) Simulations (2020).")
ggsave(paste0(Directory,"/Figure04.png"),width=29.7,height=21,units="cm")

# Figure 05: Results Comparison. ----
# ************************************************************************************************************************
# Figure 05: Results Comparison.
# ************************************************************************************************************************
Data <- readxl::read_excel("McKibbin & Fernando_2023_Data for Tables Figures and Annexures.xlsx",sheet="Figure_5_Data",skip=1)
names(Data)[1] <- "Region"
Data <- reshape2::melt(Data,id.vars=c("Region"),variable.name="Variable",value.name="Value")
Data <- tidyr::separate(Data,Variable,into=c("Metric","Year"),sep=":",remove=FALSE)
Data$Metric <- plyr::mapvalues(Data$Metric,from=sort(unique(Data$Metric)),to=c("Actual Outcome","Most Optimistic Projection","Most Pessimistic Projection"))
MetricLevels <- c("Actual Outcome","Most Optimistic Projection","Most Pessimistic Projection")
Data$Year <- plyr::mapvalues(Data$Year,from=sort(unique(Data$Year)),to=c("2020","2021"))
Data <- subset(Data,select=-c(Variable))

Plot <- 
  ggplot(data=Data[which(Data$Metric != "Actual Outcome"),],aes(x=Region,y=Value)) +
  geom_bar(aes(fill=factor(Metric,levels=c("Most Optimistic Projection","Most Pessimistic Projection"))),stat="identity",position="identity",alpha=0.5) +
  geom_point(data=Data[which(Data$Metric == "Actual Outcome"),],aes(shape=Metric),colour="black",size=3) +
  guides(shape=guide_legend(order=1),fill=guide_legend(order=1)) +
  scale_fill_manual(values=MyColors2) +
  scale_shape_manual(values=c("Actual Outcome"=16),labels="Actual Outcome") +
  Theme +
  facet_wrap(~Year,nrow=2) +
  scale_x_discrete(expand=c(0.01,0.01)) +
  scale_y_continuous(expand=c(0.01,0.01)) +
  ylab("USD Trillion") +
  theme(strip.text.x=element_text(size=11,family="serif"),panel.spacing=unit(2,"lines")) +
  labs(title=paste0("Figure 05: Comparison of Projected and Actual GDP Losses for 2020 and 2021"),caption="Source: Constructed by the Author using G-Cubed (GGG6G) Simulations (2023).")
ggsave(paste0(Directory,"/Figure05.png"),width=29.7,height=21,units="cm")
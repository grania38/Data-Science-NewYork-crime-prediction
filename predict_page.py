import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup
from folium.plugins import MarkerCluster
import sklearn

def load_model():
    with open('saved_stepss.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
le_sex = data["VIC_SEX"]
le_age = data["VIC_AGE_GROUP"]
le_prk = data["PARK_NM"]
#le_pat = data["PATROL_BORO"]
#le_prem = data["PREM_TYP_DESC"]
le_frtm = data["CMPLNT_FR_TM"]
le_brnm = data["BORO_NM"]



def show_predict():
    st.title("Crime prediction")
    st.write("All fields are required!")
    st.write("""#### We need some information""")

    vic_sex = ('M', 'F', 'E', 'D')
    vic_age_group = ('<18', '25-44', 'UNKNOWN', '45-64', '18-24', '65+')
    park_name_BRONX = ('CLAREMONT PARK', "ST. MARY'S PARK BRONX", 'FRANZ SIGEL PARK',
       'RICHMAN (ECHO) PARK', 'MOUNT HOPE PLAYGROUND', 'MACOMBS DAM PARK',
       'DEVOE PARK', 'PELHAM BAY PARK', 'JOSEPH RODMAN DRAKE PARK',
       'SLATTERY PLAYGROUND', 'WASHINGTON PARK BRONX', 'AQUEDUCT WALK',
       'PATTERSON PLAYGROUND', 'CLARK PLAYGROUND', 'ARCILLA PLAYGROUND',
       'BRONX RIVER PARKWAY', 'RAINEY PARK BRONX', 'JOYCE KILMER PARK',
       'CROTONA PARK', 'UNIVERSITY WOODS', 'WILLIAMSBRIDGE OVAL',
       'OWEN F. DOLEN PARK', 'MOSHOLU PARKWAY', 'SOUNDVIEW PARK',
       'POE PARK', 'ST. JAMES PARK', 'MULLALY PARK',
       "D'AURIA-MURPHY TRIANGLE", 'GRANT PARK', 'SAW MILL PLAYGROUND',
       'NOBLE PLAYGROUND', 'MATTHEWS MULINER PLAYGROUND', 'BRONX PARK',
       'THORPE FAMILY PLAYGROUND', 'ADMIRAL FARRAGUT PLAYGROUND',
       'SETON FALLS PARK', 'WALTON PARK', 'LORETO PLAYGROUND',
       'TREMONT PARK', 'JAMES BURKE BALLFIELD', 'STARLIGHT PARK',
       'VAN CORTLANDT PARK', 'MARBLE HILL PLAYGROUND', 'BUFANO PARK',
       'FLYNN PLAYGROUND', 'FERRY POINT PARK', 'CASERTA PLAYGROUND',
       'THE PEARLY GATES', 'REV LENA IRONS UNITY PARK',
       'PONTIAC PLAYGROUND', 'FORDHAM LANDING PLAYGROUND',
       'PELHAM PARKWAY', 'MOTT PLAYGROUND', 'ROCK GARDEN PARK',
       'CLAREMONT NEIGHBORHOOD GARDEN', 'EDENWALD PLAYGROUND',
       'VIRGINIA PARK', 'BEHAGEN PLAYGROUND', 'BERGEN TRIANGLE',
       'MAPES BALLFIELD', 'CHIEF DENNIS L. DEVLIN PARK',
       'CONCRETE PLANT PARK', 'EWEN PARK', 'HIGHBRIDGE PARK BRONX SIDE',
       'CLEOPATRA PLAYGROUND', 'SPACE TIME PLAYGROUND',
       'WEBSTER PLAYGROUND', "WASHINGTON'S WALK", 'WILLIS PLAYGROUND',
       'SETON PARK', 'AGNES HAYWOOD PLAYGROUND',
       'WATSON GLEASON PLAYGROUND', 'P.O. SERRANO PLAYGROUND',
       'HAFFEN PARK', 'BELMONT PLAYGROUND BRONX',
       'CAPTAIN RIVERA PLAYGROUND', 'FORT INDEPENDENCE PLAYGROUND',
       'MOSHOLU PLAYGROUND', 'VAN NEST PARK', 'TIFFANY PLAYGROUND',
       'VIRGINIA PLAYGROUND', 'PUGSLEY CREEK PARK', 'BRUST PARK',
       'MAGENTA PLAYGROUND', 'HUNTS POINT PLAYGROUND',
       'NELSON PLAYGROUND', 'SEABURY PARK', 'REV J POLITE PLAYGROUND',
       'UNNAMED PARK ON WEST MOUNT EDEN AVENUE', 'EASTCHESTER PLAYGROUND',
       'VIDALIA PARK', 'PULASKI PARK', 'GOBLE PLAYGROUND',
       'UNNAMED PARK ON MELROSE AVENUE', 'KELTCH PARK',
       'RAILROAD PARK BRONX', 'MELROSE PLAYGROUND', 'ALLERTON PLAYGROUND',
       'MERRIAM PLAYGROUND', 'BRYAN PARK', 'COLGATE CLOSE', 'HARRIS PARK',
       'MILL POND PARK', 'LYONS SQUARE PLAYGROUND', 'ECHO TRIANGLE',
       'JULIO CARBALLO FIELDS', "O'BRIEN OVAL", 'DREW PLAYGROUND',
       'PROSPECT PLAYGROUND', "PEOPLE'S PARK", 'HAVEMEYER PLAYGROUND',
       'LA FINCA DEL SUR COMMUNITY GARDEN', 'RIVER GARDEN',
       'CICCARONE PARK', 'CEDAR PLAYGROUND', 'GUN HILL PLAYGROUND',
       'JACKSON-FOREST COMMUNITY GARDEN')
    park_name_QUEENS = ('MARCONI PARK', 'WILLIAM F MOORE PARK',
       'FLUSHING MEADOWS CORONA PARK', 'CUNNINGHAM PARK',
       'RALPH DEMARCO PARK', 'PAUL RAIMONDA PLAYGROUND',
       'CONCH PLAYGROUND', 'BAISLEY POND PARK', 'ASTORIA PARK',
       'BEACH CHANNEL PLAYGROUND', 'MOORE HOMESTEAD PLAYGROUND',
       'FOREST PARK', 'TRAVERS PARK', 'POPPENHUSEN PARK',
       'PLAYGROUND SEVENTY FIVE', 'ALLEY POND PARK',
       'PARK OF THE AMERICAS', 'MARGARET I. CARMAN GREEN - WEEPING BEECH',
       'ALLEY ATHLETIC PLAYGROUND', 'ROCKAWAY BEACH AND BOARDWALK',
       'BAYSWATER PARK', 'MCLAUGHLIN PLAYGROUND',
       'UNNAMED PARK ON BELT PARKWAY', 'KISSENA PARK', 'BLAND PLAYGROUND',
       'TALL OAK PLAYGROUND', 'ST. ALBANS PARK', "ROSEMARY'S PLAYGROUND",
       'WILLOW LAKE PLAYGROUND', 'RUFUS KING PARK', 'ELMHURST PARK',
       'HOWARD VON DOHLEN PLAYGROUND', 'ASTORIA HEIGHTS PLAYGROUND',
       'EVERGREEN PARK', 'KISSENA CORRIDOR PARK', 'TUDOR PARK',
       'DOUGHBOY PLAZA', 'COLDEN PLAYGROUND', "ST. MICHAEL'S PLAYGROUND",
       'RUSSELL SAGE PLAYGROUND', 'REIFF PLAYGROUND', 'BROOKVILLE PARK',
       "HUNTER'S POINT SOUTH PARK", 'MACNEIL PARK', 'MAPLE PLAYGROUND',
       'CAPTAIN TILLY PARK', "POWELL'S COVE PARK", 'MONTBELLIER PARK',
       'MARIE CURIE PARK', 'MURRAY HILL PLAYGROUND',
       'GRASSMERE PLAYGROUND', 'DETECTIVE KEITH L WILLIAMS PARK',
       'JUNIPER VALLEY PARK', "FRANK D. O'CONNOR PLAYGROUND",
       'FLUSHING FIELDS', "DANIEL M. O'CONNELL PLAYGROUND",
       'HAROLD SCHNEIDERMAN PLAYGROUND', 'VAN ALST PLAYGROUND',
       'EMERALD PLAYGROUND', 'HOYT PLAYGROUND', 'EQUITY PARK',
       'SEASIDE PLAYGROUND', 'POPPENHUSEN PLAYGROUND',
       'ROY WILKINS RECREATION CENTER', 'BEACH 59TH ST PLAYGROUND',
       'FRANK GOLDEN PARK', 'HAGGERTY PARK', 'DUTCH KILLS PLAYGROUND',
       'JUNCTION PLAYGROUND', "SEAN'S PLACE", 'RAINEY PARK QUEENS',
       'PHIL "SCOOTER" RIZZUTO PARK', 'HOFFMAN PARK', 'HOLLIS PLAYGROUND',
       'FRANCIS LEWIS PLAYGROUND', 'SIMEONE PARK',
       'MIDDLE VILLAGE PLAYGROUND', 'BENNINGER PLAYGROUND',
       'POLICE OFFICER EDWARD BYRNE PARK', 'GORMAN PLAYGROUND',
       '"UNCLE" VITO E. MARANZANO GLENDALE PLAYGROUND',
       'PINOCCHIO PLAYGROUND', 'SPRINGFIELD PARK',
       'GROVER CLEVELAND PLAYGROUND', 'PLAYGROUND ONE FORTY',
       'BREININGER PARK', 'ATHENS SQUARE', 'AMERICAN TRIANGLE',
       'BOWNE PLAYGROUND', 'BOWNE PARK', 'JOSEPH AUSTIN PLAYGROUND',
       'PLAYGROUND NINETY', 'HIGHLAND PARK', 'BEACH 17 PLAYGROUND',
       'FRANCIS LEWIS PARK', 'POMONOK PLAYGROUND', 'LITTLE BAY PARK',
       'POLICE OFFICER NICHOLAS DEMUTIIS PARK', 'HARVEY PARK',
       'VETERANS GROVE', 'NORTHERN PLAYGROUND',
       'HART PLAYGROUND ON BROADWAY', 'WAYANDA PARK',
       'BEACH 9 PLAYGROUND', 'HINTON PARK', 'MURRAY PLAYGROUND',
       'WALTER WARD PLAYGROUND', 'MAFERA PARK', 'QUEENS FARM PARK',
       'LONDON PLANETREE PLAYGROUND', 'LAURELTON PLAYGROUND',
       'FREDERICK B. JUDGE PLAYGROUND',
       'LOST BATTALION HALL RECREATION CENTER', 'CROCHERON PARK',
       'FLEETWOOD TRIANGLE', 'FORT TOTTEN PARK',
       'UNNAMED PARK ON BEACH CHANNEL DRIVE & BEACH 89TH TO OLD BEACH 88TH STREET',
       'ARVERNE PLAYGROUND', 'WHITEY FORD FIELD',
       'L/CPL THOMAS P. NOONAN JR. PLAYGROUND', 'SUNRISE PLAYGROUND',
       'BAY TERRACE PLAYGROUND', "VETERAN'S SQUARE", 'BEACH CHANNEL PARK',
       'BARCLAY TRIANGLE', 'QUEENSBRIDGE PARK', 'COLLEGE POINT FIELDS',
       'MAJOR MARK PARK', 'ELECTRIC PLAYGROUND', 'GENE GRAY PLAYGROUND',
       'RAVENSWOOD PLAYGROUND', 'CHALLENGE PLAYGROUND',
       'JACKIE ROBINSON PARKWAY', 'HAMMEL PLAYGROUND',
       'PLAYGROUND SIXTY TWO LXII', 'LAURELTON PARKWAY',
       'A.R.R.O.W. FIELD HOUSE', 'CLINTONVILLE PLAYGROUND',
       'TELEPHONE PLAYGROUND')
    park_name_BROOKLYN = ('TRINITY PARK', 'BELT PARKWAY/SHORE PARKWAY', 'HIGHLAND PARK',
       'LINCOLN TERRACE / ARTHUR S. SOMERS PARK',
       "ST. JOHN'S RECREATION CENTER", 'PROSPECT PARK', 'MCCARREN PARK',
       'RED HOOK RECREATION AREA', 'OCEAN HILL PLAYGROUND',
       'PARADE GROUND', 'CALLAHAN-KELLY PLAYGROUND', 'LEIF ERICSON PARK',
       'NEHEMIAH PARK', 'RODNEY PLAYGROUND CENTER',
       'CONEY ISLAND BEACH & BOARDWALK', 'GEORGE WALKER JR. PARK',
       'FOX PLAYGROUND BROOKLYN', 'SETH LOW PLAYGROUND/ BEALIN SQUARE',
       'LT. JOSEPH PETROSINO PARK', 'MARIA HERNANDEZ PARK',
       'STERNBERG PARK', 'GRADY PLAYGROUND', 'FRIENDS FIELD',
       'JACKIE ROBINSON PLAYGROUND', 'SARATOGA BALLFIELDS',
       'SHORE PARK AND PARKWAY', 'KOLBERT PLAYGROUND', 'BETSY HEAD PARK',
       'MARINE PARK', 'FORT GREENE PARK', 'PARK SLOPE PLAYGROUND',
       'BILL BROWN PLAYGROUND', 'HERBERT VON KING PARK',
       'LINWOOD PLAYGROUND', 'CENTURY PLAYGROUND', 'SUNSET PARK',
       "OWL'S HEAD PARK", 'KELLY PARK', 'EL SHABAZZ PLAYGROUND',
       'CARROLL PARK', 'BUSHWICK INLET PARK', 'BROOKLYN BRIDGE PARK',
       'LUNA PARK', 'WILLIAM SHERIDAN PLAYGROUND', 'WEST PLAYGROUND',
       'MSGR. MCGOLRICK PARK', 'MCLAUGHLIN PARK', 'DYKER BEACH PARK',
       'PARKSIDE PLAYGROUND BROOKLYN', 'P.O. REINALDO SALGADO PLAYGROUND',
       'RAINBOW PLAYGROUND', 'STROUD PLAYGROUND',
       'ETHAN ALLEN PLAYGROUND', 'POTOMAC PLAYGROUND', 'MCKINLEY PARK',
       'MARTIN LUTHER KING JR. PLAYGROUND', 'BENSONHURST PARK',
       'GRAVESEND PARK', 'GRACE PLAYGROUND', 'SCARANGELLA PARK',
       "CARTER G. WOODSON CHILDREN'S PARK", 'SARATOGA PARK',
       'STEEPLECHASE PARK', 'JACKIE ROBINSON PARK BROOKLYN',
       'BATH BEACH PARK', 'COMMODORE BARRY PARK', 'IRVING SQUARE PARK',
       'PULASKI PLAYGROUND', "ST. ANDREW'S PLAYGROUND", 'LINDEN PARK',
       'FIDLER-WYCKOFF HOUSE PARK', 'MANHATTAN BEACH PARK', 'KAISER PARK',
       'FISH PLAYGROUND', 'CANARSIE PARK', 'CALVERT VAUX PARK',
       'GOLCONDA PLAYGROUND', 'BRIZZI PLAYGROUND',
       'RAYMOND BUSH PLAYGROUND', 'GLENWOOD PLAYGROUND',
       'SHEEPSHEAD PLAYGROUND', 'SOUTH OXFORD PARK',
       'JOHN HANCOCK PLAYGROUND', 'UNDERWOOD PARK',
       'HATTIE CARTHAN PLAYGROUND', 'TILDEN PLAYGROUND', 'THE CYCLONE',
       'WNYC TRANSMITTER PARK', 'ELTON PLAYGROUND', 'LINDSAY TRIANGLE',
       'MARLBORO PLAYGROUND', 'HARRY MAZE PLAYGROUND',
       'MARTINEZ PLAYGROUND',
       'CLASSON PLAYGROUND AT CLASSON AVENUE & LAFAYETTE AVENUE',
       'MARCY PLAYGROUND', 'HAMILTON METZ FIELD', 'BENSON PLAYGROUND',
       'BROWER PARK', 'JAIME CAMPIZ PLAYGROUND', 'CUYLER GORE PARK',
       'BRIGHTON PLAYGROUND', 'HECKSCHER PLAYGROUND',
       'ROBERT E. VENABLE PARK', 'BRIDGE PARK BROOKLYN', 'COOPER PARK',
       'GARIBALDI PLAYGROUND', 'MOUNT PROSPECT PARK',
       'J.J. BYRNE PLAYGROUND', 'HOUSTON PLAYGROUND',
       'GILBERT RAMIREZ PARK', 'FULTON PARK', 'CONTINENTAL ARMY PLAZA',
       'NICHOLAS NAQUAN HEYWARD JR. PARK',
       'COLONEL DAVID MARCUS PLAYGROUND', 'KOSCIUSZKO POOL',
       'GREENWOOD PLAYGROUND', 'DETECTIVE JOSEPH MAYROSE PARK',
       'POWELL PLAYGROUND', 'LINDOWER PARK', 'SCHENCK PLAYGROUND',
       'DR. RONALD MCNAIR PARK', 'WASHINGTON PARK BROOKLYN',
       'WINGATE PARK', 'JESSE OWENS PLAYGROUND', 'DECATUR PLAYGROUND',
       'LAFAYETTE PLAYGROUND ON LAFAYETTE AVENUE', 'OCEAN PARKWAY MALLS',
       'ELEANOR ROOSEVELT PLAYGROUND', 'TAAFFE PLAYGROUND', 'COFFEY PARK',
       'GARDEN PLAYGROUND', 'MILESTONE PARK', 'ORACLE PLAYGROUND',
       'CITY LINE PARK', 'MELLETT PLAYGROUND', 'WALT WHITMAN PARK',
       'BROWNSVILLE PLAYGROUND', 'DE HOSTOS PLAYGROUND', 'YAK PLAYGROUND',
       'BUSHWICK PLAYGROUND ON KNICKERBOCKER AVENUE', 'DOME PLAYGROUND',
       'NEWPORT PLAYGROUND', 'SIXTEEN LINDENS TRIANGLE',
       'ROLF HENRY PLAYGROUND', 'BATH PLAYGROUND', 'RODNEY PARK NORTH',
       'MARTIN LUTHER PLAYGROUND', 'NORTH 5TH STREET PIER AND PARK',
       'HOMECREST PLAYGROUND', 'CADMAN PLAZA PARK',
       'JOHN PAUL JONES PARK', 'BREVOORT PLAYGROUND',
       'CRISPUS ATTUCKS PLAYGROUND', 'PUBLIC PLACE', 'SPRING CREEK PARK',
       'BANNEKER PLAYGROUND', 'FROST PLAYGROUND',
       "MARC AND JASON'S PLAYGROUND", 'HERMAN DOLGON PLAYGROUND',
       'RAPPAPORT PLAYGROUND', "CHARLIE'S PLACE", 'DEAN PLAYGROUND',
       'BELMONT PLAYGROUND BROOKLYN', 'JOHN ALLEN PAYNE PARK',
       'SPERANDEO BROTHERS PLAYGROUND', 'CURTIS PLAYGROUND',
       'HOPE BALLFIELD', 'BROOKLYN HEIGHTS PROMENADE',
       'COLUMBUS PARK BROOKLYN', 'PINK PLAYGROUND', "GREGORY'S GARDEN",
       'CENTRAL PARK', 'MCDONALD PLAYGROUND BROOKLYN',
       'RUSSELL PEDERSEN PLAYGROUND', 'GALAPO PLAYGROUND',
       'STAR SPANGLED PLAYGROUND', 'SHEEPSHEAD BAY PIERS',
       'PAERDEGAT PARK', 'POWER PLAYGROUND', 'DIMATTINA PLAYGROUND',
       'GRAND ARMY PLAZA BROOKLYN')
    park_name_STATEISLAND = ('CPL. THOMPSON PARK', 'BLOOMINGDALE PARK',
       'MCDONALD PLAYGROUND STATEN ISLAND', 'TOMPKINSVILLE PARK',
       'SNUG HARBOR CULTURAL CENTER', 'SILVER LAKE PARK',
       'CONFERENCE HOUSE PARK', 'DE MATTI PLAYGROUND',
       'LT. LIA PLAYGROUND', 'TOTTENVILLE POOL',
       'VETERANS PARK STATEN ISLAND', 'CLOVE LAKES PARK',
       'GREAT KILLS PARK AT HILLCREST AVENUE', "WOLFE'S POND PARK",
       'MIDLAND FIELD', 'NORTH SHORE ESPLANADE', 'SOUTH BEACH WETLANDS',
       'TAPPEN PARK', 'OLD TOWN PLAYGROUND', "JENNIFER'S PLAYGROUND",
       'ARROCHAR PLAYGROUND', 'FRANKLIN D. ROOSEVELT BOARDWALK AND BEACH',
       'LYONS POOL', 'LOPEZ PLAYGROUND', 'LEMON CREEK PARK',
       'SEASIDE WILDLIFE NATURE PARK', 'JONES WOODS PARK',
       'GEN. DOUGLAS MACARTHUR PARK', 'WILLOWBROOK PARK',
       'DUGAN PLAYGROUND', 'OCEAN BREEZE PARK', 'GREAT KILLS PARK',
       'STAPLETON PLAYGROUND', 'MAHONEY PLAYGROUND', 'FRESHKILLS PARK',
       'STATEN ISLAND INDUSTRIAL PARK',
       'UNNAMED PARK ON SOUTHSIDE STATEN ISLAND EXPRESSWAY & SLOSSON AVENUE',
       'NORTHERLEIGH PARK', 'SKYLINE PLAYGROUND')
    patrol_boro = ('PATROL BORO BRONX', 'PATROL BORO QUEENS SOUTH',
       'PATROL BORO BKLYN NORTH', 'PATROL BORO MAN NORTH',
       'PATROL BORO QUEENS NORTH', 'PATROL BORO MAN SOUTH',
       'PATROL BORO BKLYN SOUTH', 'PATROL BORO STATEN ISLAND')
    #park_name_MANHATTAN = ('UNKNOWN')
    boro_name = ('BRONX', 'QUEENS', 'BROOKLYN', 'MANHATTAN', 'STATEN ISLAND')
    time = ('16:18:00', '01:30:00', '18:30:00', '04:22:00', '11:09:00',
                   '23:15:00', '16:45:00', '15:50:00', '17:40:00', '16:30:00',
                   '15:15:00', '15:00:00', '14:20:00', '14:15:00', '15:30:00',
                   '14:51:00', '04:00:00', '20:40:00', '13:50:00', '11:00:00',
                   '10:50:00', '12:00:00', '17:36:00', '20:55:00', '16:21:00',
                   '11:07:00', '05:00:00', '09:30:00', '00:42:00', '22:05:00',
                   '09:00:00', '14:40:00', '11:41:00', '08:20:00', '20:00:00',
                   '07:30:00', '17:00:00', '15:03:00', '18:45:00', '18:55:00',
                   '16:25:00', '22:00:00', '10:40:00', '16:00:00', '21:20:00',
                   '18:40:00', '15:10:00', '11:35:00', '00:01:00', '17:30:00',
                   '16:20:00', '17:04:00', '14:30:00', '12:30:00', '11:40:00',
                   '13:40:00', '16:32:00', '18:00:00', '11:30:00', '06:30:00',
                   '14:35:00', '15:20:00', '02:11:00', '01:50:00', '15:16:00',
                   '00:50:00', '14:58:00', '14:45:00', '14:31:00', '19:05:00',
                   '13:30:00', '14:26:00', '08:55:00', '09:10:00', '13:10:00',
                   '02:20:00', '03:58:00', '11:52:00', '10:00:00', '13:55:00',
                   '18:10:00', '23:25:00', '22:45:00', '09:04:00', '17:10:00',
                   '14:00:00', '12:20:00', '15:40:00', '21:05:00', '06:00:00',
                   '21:40:00', '08:00:00', '07:00:00', '09:39:00', '20:30:00',
                   '22:50:00', '01:20:00', '14:10:00', '00:15:00', '18:47:00',
                   '19:30:00', '10:28:00', '23:00:00', '20:19:00', '02:50:00',
                   '19:00:00', '22:15:00', '21:00:00', '18:31:00', '09:48:00',
                   '11:20:00', '16:35:00', '00:30:00', '04:55:00', '13:27:00',
                   '00:00:00', '08:30:00', '10:15:00', '10:30:00', '22:25:00',
                   '06:10:00', '17:14:00', '14:57:00', '16:12:00', '08:25:00',
                   '10:49:00', '17:15:00', '03:00:00', '08:15:00', '17:22:00',
                   '11:53:00', '18:50:00', '20:26:00', '01:15:00', '03:15:00',
                   '13:00:00', '20:54:00', '13:15:00', '04:50:00', '08:50:00',
                   '17:20:00', '18:35:00', '15:34:00', '15:45:00', '16:40:00',
                   '10:10:00', '17:45:00', '02:05:00', '15:05:00', '07:40:00',
                   '18:53:00', '16:15:00', '01:00:00', '12:15:00', '23:40:00',
                   '14:25:00', '10:45:00', '20:35:00', '18:20:00', '01:04:00',
                   '09:25:00', '05:30:00', '19:25:00', '01:40:00', '16:29:00',
                   '19:45:00', '16:57:00', '11:38:00', '20:16:00', '17:33:00',
                   '17:37:00', '12:07:00', '14:01:00', '14:23:00', '21:15:00',
                   '12:25:00', '13:37:00', '08:40:00', '06:52:00', '19:40:00',
                   '11:55:00', '02:00:00', '16:28:00', '09:35:00', '21:30:00',
                   '14:42:00', '17:05:00', '20:39:00', '03:35:00', '19:15:00',
                   '16:50:00', '14:50:00', '05:28:00', '09:20:00', '17:46:00',
                   '15:26:00', '17:25:00', '16:24:00', '23:10:00', '09:05:00',
                   '13:08:00', '15:35:00', '13:05:00', '20:02:00', '19:09:00',
                   '21:10:00', '23:20:00', '00:59:00', '17:56:00', '16:55:00',
                   '09:44:00', '11:25:00', '17:50:00', '21:42:00', '12:04:00',
                   '19:26:00', '13:45:00', '04:30:00', '22:35:00', '12:01:00',
                   '09:37:00', '11:50:00', '18:25:00', '21:50:00', '14:03:00',
                   '14:02:00', '19:35:00', '05:15:00', '16:47:00', '21:45:00',
                   '19:20:00', '07:15:00', '00:35:00', '18:41:00', '22:28:00',
                   '15:49:00', '14:55:00', '20:44:00', '18:43:00', '06:25:00',
                   '00:40:00', '17:35:00', '15:55:00', '16:34:00', '20:45:00',
                   '12:23:00', '21:25:00', '23:30:00', '10:09:00', '13:20:00',
                   '23:41:00', '10:55:00', '14:16:00', '19:10:00', '14:18:00',
                   '12:40:00', '12:36:00', '17:57:00', '05:55:00', '16:38:00',
                   '18:05:00', '15:56:00', '12:50:00', '12:39:00', '17:13:00',
                   '17:42:00', '10:25:00', '13:25:00', '18:21:00', '19:14:00',
                   '13:59:00', '16:37:00', '23:50:00', '16:10:00', '22:07:00',
                   '19:58:00', '13:57:00', '18:46:00', '09:45:00', '13:03:00',
                   '20:25:00', '19:44:00', '22:11:00', '20:29:00', '15:25:00',
                   '12:05:00', '11:45:00', '18:54:00', '11:05:00', '20:50:00',
                   '00:10:00', '14:54:00', '22:20:00', '10:20:00', '17:18:00',
                   '19:50:00', '01:05:00', '08:26:00', '16:36:00', '20:01:00',
                   '04:20:00', '03:50:00', '23:55:00', '08:45:00', '13:49:00',
                   '16:23:00', '20:15:00', '18:19:00', '13:28:00', '12:45:00',
                   '12:46:00', '12:10:00', '22:30:00', '21:35:00', '19:27:00',
                   '17:29:00', '09:14:00', '19:55:00', '13:02:00', '17:17:00',
                   '19:56:00', '17:06:00', '03:30:00', '03:14:00', '21:32:00',
                   '09:15:00', '12:27:00', '09:50:00', '12:55:00', '00:02:00',
                   '14:08:00', '20:49:00', '18:02:00', '16:56:00', '16:59:00',
                   '01:27:00', '22:18:00', '00:20:00', '15:54:00', '17:27:00',
                   '00:06:00', '22:40:00', '06:05:00', '08:10:00', '22:48:00',
                   '20:10:00', '14:06:00', '23:05:00', '17:53:00', '10:38:00',
                   '19:49:00', '02:30:00', '21:16:00', '15:06:00', '05:20:00',
                   '12:16:00', '10:39:00', '18:15:00', '06:58:00', '15:48:00',
                   '14:41:00', '11:32:00', '02:16:00', '17:55:00', '15:52:00',
                   '20:46:00', '14:05:00', '17:52:00', '08:35:00', '20:21:00',
                   '08:02:00', '23:08:00', '20:42:00', '20:20:00', '20:12:00',
                   '18:08:00', '04:07:00', '12:41:00', '21:34:00', '12:35:00',
                   '19:18:00', '02:40:00', '00:05:00', '22:08:00', '14:53:00',
                   '09:40:00', '06:45:00', '12:06:00', '21:27:00', '16:44:00',
                   '20:14:00', '19:31:00', '18:28:00', '19:46:00', '21:55:00',
                   '10:22:00', '00:45:00', '11:28:00', '03:20:00', '06:03:00',
                   '20:58:00', '14:33:00', '21:22:00', '02:25:00', '19:36:00',
                   '20:08:00', '19:07:00', '01:03:00', '17:31:00', '17:01:00',
                   '01:12:00', '05:11:00', '11:03:00', '04:15:00', '07:35:00',
                   '17:11:00', '21:08:00', '19:12:00', '21:31:00', '17:34:00',
                   '23:11:00', '14:46:00', '00:48:00', '16:43:00', '11:15:00',
                   '07:20:00', '00:51:00', '18:32:00', '07:50:00', '21:52:00',
                   '04:40:00', '15:44:00', '13:33:00', '19:02:00', '07:08:00',
                   '16:51:00', '13:11:00', '18:38:00', '22:59:00', '02:55:00',
                   '19:03:00', '21:51:00', '20:18:00', '09:28:00', '08:51:00',
                   '04:34:00', '07:45:00', '06:50:00', '02:10:00', '06:35:00',
                   '14:32:00', '11:10:00', '21:37:00', '14:21:00', '15:38:00',
                   '03:45:00', '05:05:00', '23:09:00', '23:14:00', '21:23:00',
                   '00:25:00', '11:02:00', '07:18:00', '17:28:00', '00:24:00',
                   '19:01:00', '01:36:00', '18:23:00', '21:39:00', '09:57:00',
                   '14:19:00', '14:37:00', '21:17:00', '20:09:00', '13:12:00',
                   '14:14:00', '08:34:00', '08:05:00', '23:46:00', '20:06:00',
                   '20:04:00', '13:14:00', '03:55:00', '20:24:00', '01:38:00',
                   '13:39:00', '19:43:00', '17:43:00', '04:05:00', '22:17:00',
                   '06:55:00', '10:32:00', '01:55:00', '21:28:00', '00:32:00',
                   '05:10:00', '15:29:00', '14:43:00', '11:14:00', '19:37:00',
                   '21:33:00', '16:19:00', '18:16:00', '21:24:00', '07:55:00',
                   '19:59:00', '22:24:00', '19:47:00', '10:23:00', '18:49:00',
                   '00:03:00', '17:08:00', '23:45:00', '18:18:00', '18:14:00',
                   '01:22:00', '01:10:00', '02:34:00', '00:56:00', '03:26:00',
                   '19:06:00', '21:04:00', '15:57:00', '01:11:00', '01:45:00',
                   '02:15:00', '19:16:00', '14:34:00', '20:38:00', '22:19:00',
                   '03:46:00', '09:53:00', '20:13:00', '14:17:00', '02:28:00',
                   '22:10:00', '17:49:00', '12:56:00', '10:53:00', '15:21:00',
                   '03:25:00', '19:38:00', '10:35:00', '19:52:00', '19:28:00',
                   '12:22:00', '18:11:00', '10:26:00', '06:36:00', '18:17:00',
                   '16:58:00', '14:12:00', '13:19:00', '18:52:00', '19:32:00',
                   '13:38:00', '04:10:00', '03:43:00', '20:31:00', '17:48:00',
                   '00:12:00', '21:47:00', '00:27:00', '13:35:00', '15:43:00',
                   '04:54:00', '09:29:00', '01:18:00', '20:17:00', '10:05:00',
                   '11:34:00', '03:10:00', '12:59:00', '17:09:00', '13:16:00',
                   '12:51:00', '21:38:00', '20:05:00', '15:32:00', '16:08:00',
                   '06:20:00', '14:36:00', '12:09:00', '12:08:00', '17:54:00',
                   '21:19:00', '03:42:00', '14:28:00', '16:49:00', '03:52:00',
                   '20:47:00', '18:22:00', '08:28:00', '10:27:00', '23:42:00',
                   '01:25:00', '13:44:00', '15:12:00', '13:32:00', '13:41:00',
                   '15:22:00', '17:51:00', '21:59:00', '16:27:00', '11:19:00',
                   '08:56:00', '03:32:00', '19:54:00', '23:24:00', '09:46:00',
                   '12:47:00', '18:24:00', '16:52:00', '02:45:00', '15:23:00',
                   '17:38:00', '08:14:00', '14:24:00', '17:32:00', '13:47:00',
                   '20:53:00', '18:37:00', '21:02:00', '11:18:00', '18:48:00',
                   '12:28:00', '13:48:00', '06:15:00', '18:57:00', '17:24:00',
                   '05:40:00', '13:42:00', '19:57:00', '12:18:00', '20:03:00',
                   '23:39:00', '10:46:00', '09:55:00', '22:34:00', '12:58:00',
                   '21:11:00', '12:34:00', '18:06:00', '07:28:00', '15:53:00')
    

    vs = st.selectbox("Select your gender" , vic_sex)
    va = st.selectbox("Select your age group" , vic_age_group)
    bn = st.selectbox("Select the borough" , boro_name)
    if (bn == 'BRONX'):
        prk = st.selectbox("Select park name" , park_name_BRONX)
        loc=[40.8448 , -73.8648]
    if (bn == 'BROOKLYN'):
        prk = st.selectbox("Select park name" , park_name_BROOKLYN)
        loc=[40.6782 , -73.9442]
    if(bn == 'QUEENS'):
        prk = st.selectbox("Select park name" , park_name_QUEENS)
        loc=[40.7282 , -73.7949]
    if(bn == 'STATEN ISLAND'):
        prk = st.selectbox("Select park name" , park_name_STATEISLAND)
        loc=[40.5795 , -73.1502]
    else:
        loc=[40.7831 , -73.9712]
#         prk = st.selectbox("Select park name" , park_name_MANHATTAN) , (bn == 'MANHATTAN'):


    tm = st.selectbox("Select nearest time" , time )

#     time = st.text_input("Enter time HH:MM:SS")
#     if len(time) ==0 :
#             st.error("Please enter a valid time")



    #Create the base Map
    m = folium.Map(location=loc, tiles='OpenStreetMap', zoom_start=12)

    folium.features.GeoJson('NYCHA PSA (Police Service Areas).geojson',
        name="Police stations", popup=folium.features.GeoJsonPopup(fields=["address"])).add_to(m)
    #folium.features.GeoJson('NYPD Sectors.geojson',
    #        name="NYPD Sectors", popup=folium.features.GeoJsonPopup(fields=["sector"])).add_to(m)

    #folium.features.GeoJson('Police Precincts.geojson',
     #           name="Police Percincts", popup=folium.features.GeoJsonPopup(fields=["precinct"])).add_to(m)

    m.add_child(folium.LatLngPopup())
    folium_static(m)
    #st.write("""#### Referring to the map, please enter:""")

#     lat = st.text_input("Latitude")
#     lon = st.text_input("Longitude")


    ok = st.button("Calculate your result")
    if ok:
        x = np.array([[vs , va , tm , bn , prk]])  #, lat , lon
        x[:,0] = le_sex.transform(x[:,0])
        x[:,1] = le_age.transform(x[:,1])
        x[:,2] = le_frtm.transform(x[:,2])
        x[:,3] = le_brnm.transform(x[:,3])
        x[:,4] = le_prk.transform(x[:,4])
#         x[:,5] = le_lat.transform(x[:,5])
#         x[:,6] = le_lon.transform(x[:,6])
        x = x.astype(float)
        crime_pred = model_loaded.predict(x)
        pred_prob = model_loaded.predict_proba(x)
        #st.subheader(f"The crime prediction is {crime_pred[0]:.2f}")
        st.subheader(f"The FELONY prediction probability at these conditions is {pred_prob[0][0]:.2f}")
        st.subheader(f"The MISDEMEANOR prediction probability at these conditions is {pred_prob[0][1]:.2f}")
        st.subheader(f"The VIOLATION prediction probability at these conditions is {pred_prob[0][2]:.2f}")
        if pred_prob[0][0] > 0.5 :
            st.subheader("Be careful !")
        if pred_prob[0][1] > 0.5 :
            st.subheader("Be careful !")
        elif pred_prob[0][2] > 0.5 :
            st.subheader("Be careful !")

        
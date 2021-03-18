# Support and confidence are values to specify the sensitivity of the algorithm that you're running.
# Use it together with this URL to do calls to the public API.
# There is a pyspotlight wrapper, or you can do pure API calls (there is atleast on tutorial online)
AnnotationURL = "https://api.dbpedia-spotlight.org/en/annotate"


# Properties that you want to return from the dbpedia entries can be defined as dbp%3AbirthPlace | %3 is a seperator. All the properties can be found if you look at a page like
# https://dbpedia.org/page/Sonic_the_Hedgehog_(1991_video_game)
# I.e dbo%3gameArtist would return Rieko_Kodama if you try to call the Entity for the sonic game with that parameter ^^
# You need to set the accept headers to JSON to use the API.
DBpedia_URL = f"http://vmdbpedia.informatik.uni-leipzig.de:8080/api/1.0.0/values?entities={person}&property={parameter1}property={parameter2}&pretty=NONE&limit=100&offset=0&key=1234&oldVersion=true"

# Playground to find the right parameters http://vmdbpedia.informatik.uni-leipzig.de:8080/swagger-ui.html#!/Request32Types/valueRequestUsingGET

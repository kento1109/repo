require 'sinatra'
require 'sinatra/reloader' 
require 'neography'
require '../cyper.rb'

get '/' do
  erb :inputs
end

post '/confirm' do
  @name = params[:name]
  erb :confirm
end

get '/time' do |name|
  code = "<%= Time.now %>"
  erb code
end

get '/home' do
  @mon = { "Jan" => 1, "Feb" => 2, "Mar" => 3 }
  erb :home
end

post '/edit' do
  body = request.body.read
  @neo = Neography::Rest.new({:authentication => 'basic', :username => "neo4j", :password => "neo4j"})
  # @neo = Neography::Rest.new("http://neo4j:neo4j@localhost:7474")

  names = @neo.execute_query("MATCH (n:PERSON) RETURN n.name")
 
  if body == ''
    status 400
  else
    json_list = []
    names["data"].each do |name|
      json_list.push({"name":name})
    end
      
    json_list.to_json
  end
end

__END__

@@inputs
<html>
  <head>
  </head>
  <body>
    <form action="confirm" method="POST">
    <input type="text" name="name" value="">
    <input type="submit" value="submit">  
    </form>
  </body>
</html>

@@confirm
<html>
  <head>
  </head>
  <body>
    Hello <%=@name%>
    </form>
  </body>
</html>

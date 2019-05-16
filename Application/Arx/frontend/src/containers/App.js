import React, { Component } from "react";
import ReactDOM from "react-dom";
import { HashRouter, Route, Switch } from 'react-router-dom';

import FrontPage from "./FrontPage";
import ContentPage from "./ContentPage";
import Home from "./Home";
import FastClassify from "./FastClassify";
import PlayerArticles from "./PlayerArticles";
import PlayerInfo from "./PlayerInfo";

class App extends Component {
  render(){
    return (
      <HashRouter>
        <Switch>
          <Route path="/article/:article_id" component={ContentPage} />
          <Route path="/articles" component={FrontPage} />
          <Route path="/player_info" component={PlayerInfo} />
          <Route path="/fast_classify" component={FastClassify} />
          <Route path="/player_articles" component={PlayerArticles} />
          <Route component={Home} />
        </Switch>
      </HashRouter>
    );
  }
}

export default App;

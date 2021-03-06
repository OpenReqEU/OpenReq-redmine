# Redmine qualification plugin

Redmine plugin to automatically fill Redmine issue fields by fetching endpoints.
This version is a fork for the OpenReq project.

## Installation

- Clone or download this repository into your Redmine's plugins folder
- restart the Redmine server
- configure the plugin in the admin panel

## Configuration

- Go to /settings/plugin/qualification and configure the plugin
  - Map the fields you want to predict to http endpoints

## Endpoint requirements

The plugin currently supports GET HTTP request only. The plugin appends to the endpoint the user query which corresponds to "?t=issue_title&b=issue_description".

The endpoint must return a JSON response which returns an object with the following structure `{ topScoringIntent: { intent: value } }`.
